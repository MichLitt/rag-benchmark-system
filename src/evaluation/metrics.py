import re
from collections import Counter


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> bool:
    return _normalize_text(prediction) == _normalize_text(gold)


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    num_same = sum((pred_counter & gold_counter).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_any(prediction: str, gold_answers: list[str]) -> bool:
    if not gold_answers:
        return False
    return any(exact_match(prediction, gold) for gold in gold_answers)


def max_f1_score(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(f1_score(prediction, gold) for gold in gold_answers)


def recall_at_k(retrieved_doc_ids: list[str], gold_doc_id: str, k: int) -> float:
    return 1.0 if gold_doc_id in retrieved_doc_ids[:k] else 0.0
