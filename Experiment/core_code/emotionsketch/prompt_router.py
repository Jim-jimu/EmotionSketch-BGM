from __future__ import annotations

import json
import math
import pickle
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


LABELS = ["Q1", "Q2", "Q3", "Q4"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_label(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().upper()
    if text in LABEL_TO_ID:
        return LABEL_TO_ID[text]
    if text.isdigit():
        return int(text)
    raise ValueError(f"Unsupported label: {value!r}")


def char_ngrams(text: str, min_n: int = 1, max_n: int = 3) -> list[str]:
    chars = [char for char in text.strip().lower() if not char.isspace()]
    feats: list[str] = []
    for n in range(min_n, max_n + 1):
        for index in range(0, max(0, len(chars) - n + 1)):
            feats.append("".join(chars[index : index + n]))
    return feats


def featurize(text: str, vocab: dict[str, int]) -> dict[int, float]:
    counts = Counter(char_ngrams(text))
    return {vocab[token]: float(count) for token, count in counts.items() if token in vocab}


def build_vocab(rows: list[dict], min_count: int = 1, max_features: int = 4000) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(set(char_ngrams(row["text"])))
    tokens = [token for token, count in counts.most_common() if count >= min_count]
    return {token: index for index, token in enumerate(tokens[:max_features])}


def softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


@dataclass
class PromptQRouter:
    vocab: dict[str, int]
    weights: list[list[float]]
    bias: list[float]

    def predict_proba(self, text: str) -> list[float]:
        features = featurize(text, self.vocab)
        logits = self.bias[:]
        for label_id in range(len(LABELS)):
            row = self.weights[label_id]
            logits[label_id] += sum(row[index] * value for index, value in features.items())
        return softmax(logits)

    def predict(self, text: str) -> tuple[str, list[float]]:
        probs = self.predict_proba(text)
        label_id = max(range(len(probs)), key=lambda index: probs[index])
        return LABELS[label_id], probs

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"vocab": self.vocab, "weights": self.weights, "bias": self.bias}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "PromptQRouter":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        return cls(vocab=payload["vocab"], weights=payload["weights"], bias=payload["bias"])


def train_router(
    rows: list[dict],
    epochs: int = 80,
    lr: float = 0.08,
    l2: float = 1e-5,
    seed: int = 20260510,
) -> PromptQRouter:
    rng = random.Random(seed)
    vocab = build_vocab(rows)
    weights = [[0.0 for _ in vocab] for _ in LABELS]
    bias = [0.0 for _ in LABELS]
    indexed = [(featurize(row["text"], vocab), normalize_label(row["label"])) for row in rows]

    for _epoch in range(epochs):
        rng.shuffle(indexed)
        for features, target in indexed:
            logits = bias[:]
            for label_id in range(len(LABELS)):
                row = weights[label_id]
                logits[label_id] += sum(row[index] * value for index, value in features.items())
            probs = softmax(logits)
            for label_id in range(len(LABELS)):
                grad = probs[label_id] - (1.0 if label_id == target else 0.0)
                bias[label_id] -= lr * grad
                row = weights[label_id]
                for index, value in features.items():
                    row[index] -= lr * (grad * value + l2 * row[index])
    return PromptQRouter(vocab=vocab, weights=weights, bias=bias)


def macro_f1(gold: list[int], pred: list[int]) -> float:
    scores = []
    for label_id in range(len(LABELS)):
        tp = sum(1 for g, p in zip(gold, pred) if g == label_id and p == label_id)
        fp = sum(1 for g, p in zip(gold, pred) if g != label_id and p == label_id)
        fn = sum(1 for g, p in zip(gold, pred) if g == label_id and p != label_id)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def evaluate_router(router: PromptQRouter, rows: list[dict]) -> dict:
    gold: list[int] = []
    pred: list[int] = []
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    predictions = []
    for row in rows:
        label, probs = router.predict(row["text"])
        gold_id = normalize_label(row["label"])
        pred_id = LABEL_TO_ID[label]
        gold.append(gold_id)
        pred.append(pred_id)
        confusion[gold_id][pred_id] += 1
        predictions.append(
            {
                "id": row.get("id"),
                "text": row["text"],
                "gold": LABELS[gold_id],
                "pred": label,
                "probs": {LABELS[index]: round(prob, 6) for index, prob in enumerate(probs)},
            }
        )
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return {
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "macro_f1": macro_f1(gold, pred) if rows else 0.0,
        "confusion": confusion,
        "predictions": predictions,
    }


def rule_predict(text: str) -> str:
    rules = {
        "Q1": ["开心", "明亮", "激动", "胜利", "欢快", "热烈", "兴奋", "庆祝", "昂扬", "活力"],
        "Q2": ["紧张", "危险", "压迫", "愤怒", "焦虑", "追逐", "冲突", "急促", "悬疑", "恐惧"],
        "Q3": ["悲伤", "孤独", "低落", "沉重", "失落", "忧郁", "哀伤", "寒冷", "空旷", "疲惫"],
        "Q4": ["平静", "温柔", "治愈", "放松", "安宁", "舒缓", "柔和", "夜景", "怀旧", "安心"],
    }
    scores = {label: 0 for label in LABELS}
    for label, keywords in rules.items():
        scores[label] = sum(1 for keyword in keywords if keyword in text)
    return max(LABELS, key=lambda label: (scores[label], -LABELS.index(label)))


def evaluate_rule(rows: list[dict]) -> dict:
    gold = [normalize_label(row["label"]) for row in rows]
    pred = [LABEL_TO_ID[rule_predict(row["text"])] for row in rows]
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    for g, p in zip(gold, pred):
        confusion[g][p] += 1
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return {
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "macro_f1": macro_f1(gold, pred) if rows else 0.0,
        "confusion": confusion,
    }
