from __future__ import annotations

from collections.abc import Iterable


def recall(returned: Iterable[str], positives: Iterable[str]) -> float:
    returned_set = set(returned)
    positive_set = set(positives)
    if not positive_set:
        return 0.0
    return len(returned_set & positive_set) / len(positive_set)


def precision(returned: Iterable[str], positives: Iterable[str]) -> float:
    returned_set = set(returned)
    positive_set = set(positives)
    if not returned_set:
        return 0.0
    return len(returned_set & positive_set) / len(returned_set)


def f1(returned: Iterable[str], positives: Iterable[str]) -> float:
    p = precision(returned, positives)
    r = recall(returned, positives)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def final_answer_found(output_texts: Iterable[str], answer: str) -> bool:
    needle = answer.casefold()
    return any(needle in text.casefold() for text in output_texts)


def trajectory_recall(encountered: Iterable[str], positives: Iterable[str]) -> float:
    return recall(encountered, positives)
