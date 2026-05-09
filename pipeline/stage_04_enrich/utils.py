"""
Utilities for Pipeline 2 - Cross-lingual Knowledge Units.
"""

import json
import re
from pathlib import Path
from typing import Iterable


EN_KEYWORDS = (
    "reinforcement learning",
    "machine learning",
    "language model",
    "self-attention",
    "multi-head",
    "learning rate",
    "fine-tuning",
    "fine tuning",
    "backpropagation",
    "tokenization",
    "transformer",
    "attention",
    "optimization",
    "validation",
    "regularization",
    "hyperparameter",
    "tokenizer",
    "vocabulary",
    "embedding",
    "convolutional",
    "recurrent",
    "probability",
    "inference",
    "gradient",
    "activation",
    "training",
    "learning",
    "network",
    "neural",
    "vector",
    "tensor",
    "softmax",
    "logit",
    "reward",
    "policy",
    "agent",
    "action",
    "encoder",
    "decoder",
    "token",
    "gpt",
    "bert",
    "llm",
)


VI_KEYWORDS = {
    "reinforcement learning": "hoc tang cuong",
    "machine learning": "hoc may",
    "language model": "mo hinh ngon ngu",
    "self-attention": "tu chu y",
    "multi-head": "nhieu dau",
    "learning rate": "toc do hoc",
    "fine-tuning": "tinh chinh",
    "fine tuning": "tinh chinh",
    "backpropagation": "lan truyen nguoc",
    "tokenization": "tach token",
    "transformer": "transformer",
    "attention": "chu y",
    "optimization": "toi uu",
    "validation": "kiem dinh",
    "regularization": "chinh quy",
    "hyperparameter": "sieu tham so",
    "tokenizer": "bo tach token",
    "vocabulary": "tu vung",
    "embedding": "nhung",
    "convolutional": "tich chap",
    "recurrent": "hoi quy",
    "probability": "xac suat",
    "inference": "suy luan",
    "gradient": "dao ham",
    "activation": "kich hoat",
    "training": "huan luyen",
    "learning": "hoc",
    "network": "mang",
    "neural": "neural",
    "vector": "vec to",
    "tensor": "tensor",
    "softmax": "softmax",
    "logit": "logit",
    "reward": "phan thuong",
    "policy": "chinh sach",
    "agent": "tac tu",
    "action": "hanh dong",
    "encoder": "bo ma hoa",
    "decoder": "bo giai ma",
    "token": "token",
    "gpt": "gpt",
    "bert": "bert",
    "llm": "llm",
}


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def write_jsonl(records: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _contains_keyword(text_lower: str, keyword: str) -> bool:
    pattern = rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"
    return re.search(pattern, text_lower) is not None


def _keyword_tokens(keyword: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", keyword.lower()))


def extract_keywords(
    text: str, max_keywords: int = 6, min_length: int = 3
) -> list[str]:
    text_lower = text.lower()
    found: list[str] = []

    for keyword in EN_KEYWORDS:
        if len(keyword) < min_length:
            continue
        if not _contains_keyword(text_lower, keyword):
            continue

        keyword_token_set = _keyword_tokens(keyword)
        is_redundant = any(keyword_token_set.issubset(_keyword_tokens(existing)) for existing in found)
        if not is_redundant:
            found.append(keyword)

        if len(found) >= max_keywords:
            break

    return found


def get_vi_keywords(en_keywords: list[str]) -> list[str]:
    vi_keywords: list[str] = []
    seen = set()
    for keyword in en_keywords:
        translated = VI_KEYWORDS.get(keyword)
        if translated and translated not in seen:
            vi_keywords.append(translated)
            seen.add(translated)
    return vi_keywords


def detect_topic(text: str) -> str:
    text_lower = text.lower()
    topic_patterns = {
        "Transformer/Attention": [
            "transformer",
            "attention",
            "self-attention",
            "multi-head",
            "encoder",
            "decoder",
        ],
        "Training/Optimization": [
            "training",
            "learning rate",
            "optimizer",
            "gradient",
            "loss",
            "backprop",
            "fine-tuning",
            "fine tuning",
        ],
        "NLP/Language Models": [
            "language model",
            "token",
            "tokenizer",
            "bert",
            "gpt",
            "embedding",
            "vocabulary",
        ],
        "Reinforcement Learning": [
            "reward",
            "policy",
            "agent",
            "action",
            "environment",
        ],
        "Neural Networks": ["neural", "network", "layer", "weight", "bias"],
        "Computer Vision": ["image", "pixel", "convolution", "cnn", "feature"],
    }

    scores = {
        topic: sum(1 for keyword in keywords if _contains_keyword(text_lower, keyword))
        for topic, keywords in topic_patterns.items()
    }

    best_topic = max(scores, key=scores.get, default="General AI/ML")
    return best_topic if scores.get(best_topic, 0) > 0 else "General AI/ML"


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def token_count(text: str) -> int:
    words = word_count(text)
    return int(round(words * 1.28)) if words else 0


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
