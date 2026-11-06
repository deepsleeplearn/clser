#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-11-06
# @Author : GJason
# @File : metrics_text2text.py

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from clser.tasks.text2text.preprocess import tokenize_for_metric


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _f1(overlap: int, pred_len: int, ref_len: int) -> float:
    if pred_len == 0 or ref_len == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_len
    recall = overlap / ref_len
    return 2 * precision * recall / (precision + recall)


def rouge_n(pred: Sequence[str], ref: Sequence[str], n: int) -> float:
    pred_ngrams, ref_ngrams = _ngrams(pred, n), _ngrams(ref, n)
    overlap = sum((pred_ngrams & ref_ngrams).values())
    return _f1(overlap, sum(pred_ngrams.values()), sum(ref_ngrams.values()))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: Sequence[str], ref: Sequence[str]) -> float:
    return _f1(_lcs_length(pred, ref), len(pred), len(ref))


def corpus_bleu(
    preds: Sequence[Sequence[str]],
    refs: Sequence[Sequence[str]],
    max_order: int = 4,
    smooth: bool = False,
) -> float:
    matches = [0] * max_order
    possible = [0] * max_order
    pred_length = 0
    ref_length = 0

    for pred, ref in zip(preds, refs):
        pred_length += len(pred)
        ref_length += len(ref)
        for n in range(1, max_order + 1):
            pred_ngrams, ref_ngrams = _ngrams(pred, n), _ngrams(ref, n)
            matches[n - 1] += sum((pred_ngrams & ref_ngrams).values())
            possible[n - 1] += max(len(pred) - n + 1, 0)

    precisions = []
    for n in range(max_order):
        if smooth:
            precisions.append((matches[n] + 1.0) / (possible[n] + 1.0))
        elif possible[n] > 0:
            precisions.append(matches[n] / possible[n])
        else:
            precisions.append(0.0)

    if min(precisions) <= 0:
        return 0.0

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    if pred_length == 0:
        return 0.0
    ratio = pred_length / ref_length if ref_length > 0 else 0.0
    brevity_penalty = 1.0 if ratio > 1.0 else (math.exp(1 - 1 / ratio) if ratio > 0 else 0.0)
    return geo_mean * brevity_penalty


def compute_text_metrics(
    predictions: List[str],
    references: List[str],
    measures: Sequence[str],
    tokenize_mode: str = "char",
    tokenizer: Optional[Any] = None,
) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have the same length, got {len(predictions)} vs {len(references)}"
        )
    if not predictions:
        return {measure: 0.0 for measure in measures}

    pred_tokens = [tokenize_for_metric(p, tokenize_mode, tokenizer) for p in predictions]
    ref_tokens = [tokenize_for_metric(r, tokenize_mode, tokenizer) for r in references]

    result: Dict[str, float] = {}
    for measure in measures:
        if measure == "rouge1":
            scores = [rouge_n(p, r, 1) for p, r in zip(pred_tokens, ref_tokens)]
        elif measure == "rouge2":
            scores = [rouge_n(p, r, 2) for p, r in zip(pred_tokens, ref_tokens)]
        elif measure == "rougeL":
            scores = [rouge_l(p, r) for p, r in zip(pred_tokens, ref_tokens)]
        elif measure == "bleu":
            result[measure] = corpus_bleu(pred_tokens, ref_tokens)
            continue
        else:
            raise ValueError(f"unknown measure: {measure}")
        result[measure] = sum(scores) / len(scores)

    return result
