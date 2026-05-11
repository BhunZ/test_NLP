"""
Language detection module for RAG backend service.

Three-stage hybrid detection pipeline:
1. Diacritic detection (fast, high confidence)
2. Token-based heuristic scoring
3. FastText fallback (only for uncertain cases)
"""

import re
from dataclasses import dataclass
from typing import Final

# =============================================================================
# STAGE 1: Precompiled Regexes (compile ONCE at module import)
# =============================================================================

# Vietnamese diacritics pattern - used for Stage 1
DIACRITIC_RE: Final[re.Pattern] = re.compile(
    r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]',
    re.IGNORECASE
)

# Tokenizer: properly handle contractions and Vietnamese
TOKEN_RE: Final[re.Pattern] = re.compile(
    r"[a-zA-ZÀ-ỹ]+(?:'[a-zA-ZÀ-ỹ]+)?",
    re.IGNORECASE
)

# =============================================================================
# STAGE 2: Precomputed Token Sets (use frozensets for immutability + fast lookup)
# =============================================================================

# Strong English indicators - high weight words that clearly indicate English
# These words alone can override weak Vietnamese signals
ENGLISH_STRONG: Final[frozenset[str]] = frozenset({
    'what', 'how', 'why', 'when', 'where', 'which', 'who', 'explain',
    'describe', 'define', 'tell', 'show', 'give', 'is', 'are', 'was', 'were',
    'does', 'did', 'can', 'the', 'and', 'or', 'but', 'for', 'with', 'this',
    'that', 'it', 'from', 'have', 'has', 'been', 'will', 'would', 'could',
    'should', 'about', 'into', 'over', 'after', 'their', 'there', 'these',
    'those', 'some', 'any', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'such', 'not', 'only', 'same', 'so', 'than', 'too',
    'very', 'just', 'make', 'take', 'come', 'see', 'know', 'think', 'want',
    'use', 'find', 'try', 'call', 'ask', 'work', 'seem', 'feel', 'leave',
    'put', 'keep', 'let', 'begin', 'become', 'hear', 'play', 'run', 'move',
    'live', 'believe', 'bring', 'happen', 'write', 'provide', 'sit', 'stand',
    'lose', 'meet', 'continue', 'set', 'learn', 'change', 'lead', 'understand',
    # Technical terms common in Stanford NLP courses
    'mechanism', 'function', 'works', 'difference', 'different', 'between',
    'compare', 'algorithm', 'model', 'training', 'inference', 'embedding',
    'network', 'layer', 'parameter', 'gradient', 'optimizer', 'loss', 'accuracy'
})

# Medium English indicators - common but not decisive
ENGLISH_MEDIUM: Final[frozenset[str]] = frozenset({
    'into', 'over', 'under', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'while', 'although', 'though', 'because',
    'since', 'until', 'unless', 'however', 'therefore', 'otherwise'
})

# Vietnamese word sets (without diacritics - common in user typing)
# These are STRONG Vietnamese indicators when found together
# NOT single letters like 'co', 'con', 'do', 'long' (too ambiguous)
VIETNAMESE_STRONG: Final[frozenset[str]] = frozenset({
    # Question patterns (very strong signal)
    'la gi', 'la gi', 'cay gi', 'gi',  # là gì, cái gì, gì
    'nao', 'nao',  # nào (which/what)
    'dau', 'dau',  # đâu (where)
    'sao', 'sao',  # sao (why/how)
    'nao', 'nao',  # nào
    'tai sao', 'tai sao', 'vi sao', 'vi sao',  # tại sao, vì sao
    
    # Common Vietnamese words (2+ chars, unambiguous)
    'khong', 'ko',  # không
    'duoc', 'dc',  # được  
    'trong',  # trong
    'voi', 'voi',  # với
    'den', 'den',  # đến
    'tren', 'tren',  # trên
    'duoi', 'duoi',  # dưới
    'cua', 'cua',  # của
    'cac', 'cac',  # các
    'cho', 'cho',  # cho
    'nhung', 'nhung',  # nhưng
    'neu', 'neu',  # nếu
    'khi', 'khi',  # khi
    'con', 'con',  # còn (but also English - handled by context)
    'the', 'the',  # thế (in "còn thế" context)
    'co', 'co',  # có (very common in Vietnamese)
    'cung', 'cung',  # cũng
    'nay', 'nay',  # này
    'kia', 'kia',  # kia
    'lam', 'lam',  # làm
    'duoc', 'duoc',  # được
    
    # Technical Vietnamese (without diacritics)
    'cach',  # cách (method)
    'cong thuc', 'cong thuc',  # công thức
    'hoat dong', 'hoat dong',  # hoạt động
    'hoat', 'hoat',  # hoạt
    'dong', 'dong',  # động
    'giai thich', 'giai thich',  # giải thích
    'giai', 'giai',  # giải
    'thich', 'thich',  # thích
    'thcih', 'thcih',  # typo for thích
    'nghia', 'nghia',  # nghĩa
    'hieu', 'hieu',  # hiểu
    'phan', 'phan',  # phần
    'tong hop', 'tong hop',  # tổng hợp
    'chi tiet', 'chi tiet',  # chi tiết
    'vi du', 'vi du',  # ví dụ
    'tuong tu', 'tuong tu',  # tương tự
    'co ban', 'co ban',  # cơ bản
    'quan trong', 'quan trong',  # quan trọng
    'ung dung', 'ung dung',  # ứng dụng
    'buoc', 'buoc',  # bước
    'quy trinh', 'quy trinh',  # quy trình
    'thuc hien', 'thuc hien',  # thực hiện
    'xay dung', 'xay dung',  # xây dựng
    
    # Typo variants
    'laf',  # typo for "la" (là)
    'gif',  # typo for "gi" (gì)
    'gifs',  # typo for "gi" (gì)
    'dogn',  # typo for "dong" (động)
    'sai',  # typo for "sao" (sao)
    'tai',  # part of "tai sao"
    'nhu', 'nhu',  # như
})

# Vietnamese question patterns (regex-based, not token-based)
# These are specific word sequences that strongly indicate Vietnamese
# Includes patterns for common typos (e.g., "laf" -> "la", "gifs" -> "gi")
VIETNAMESE_PATTERNS: Final[list[tuple[re.Pattern, float]]] = [
    # là gì patterns (strongest indicator)
    (re.compile(r'\bla\s+gi\b', re.IGNORECASE), 1.0),
    
    # là + short word (catches typos like "laf gifs" -> "la gi")
    # This pattern will match "la" followed by 1-4 letters (handles "laf", "la", etc.)
    (re.compile(r'\bla\s+[a-z]{1,5}\b', re.IGNORECASE), 0.6),
    
    # là + anything (more flexible - captures "la gif" as "la gi")
    (re.compile(r'\bla\s+\w+', re.IGNORECASE), 0.5),
    
    # có/không patterns
    (re.compile(r'\bcó\s+không\b', re.IGNORECASE), 0.9),
    (re.compile(r'\bco\s+khong\b', re.IGNORECASE), 0.8),
    (re.compile(r'\bco\s+\w+', re.IGNORECASE), 0.4),  # co + anything (very common Vietnamese)
    
    # tại sao / vì sao patterns (why)
    (re.compile(r'\btai\s*s(a)?o\b', re.IGNORECASE), 0.9),
    (re.compile(r'\bvi\s*sao\b', re.IGNORECASE), 0.9),
    
    # hoạt động patterns (how it works)
    (re.compile(r'\bhoat\s*d(on)?g\b', re.IGNORECASE), 0.8),
    (re.compile(r'\bhoạt\s*động\b', re.IGNORECASE), 0.9),
    
    # giải thích patterns (explain)
    (re.compile(r'\bgiai\s*t(h)?ich\b', re.IGNORECASE), 0.8),
    (re.compile(r'\bgiải\s*thích\b', re.IGNORECASE), 0.9),
    
    # Technical terms without diacritics (strong signal for NLP domain)
    (re.compile(r'\bcông\s*thức\b', re.IGNORECASE), 0.9),
    (re.compile(r'\bcong\s*thuc\b', re.IGNORECASE), 0.8),
    
    # Common question patterns
    (re.compile(r'\bla\s+\w{1,4}\b', re.IGNORECASE), 0.7),  # là + short word
    (re.compile(r'\bcó\s+\w+', re.IGNORECASE), 0.5),  # có + anything
    
    # "nao" patterns (nào - what/which)
    (re.compile(r'\bnao\b', re.IGNORECASE), 0.7),
    
    # nhu nao patterns (how - as in "như thế nào")
    (re.compile(r'\bnhu\s+nao\b', re.IGNORECASE), 0.8),
]

# =============================================================================
# STAGE 3: FastText Configuration
# =============================================================================

# Confidence thresholds
HIGH_CONFIDENCE: Final[float] = 0.7
LOW_CONFIDENCE: Final[float] = 0.5

# FastText model path (will be set if available)
FASTTEXT_MODEL_PATH: str | None = None
_fasttext_model = None


def _load_fasttext_model() -> tuple | None:
    """Lazy load FastText model only when needed."""
    global _fasttext_model, FASTTEXT_MODEL_PATH
    
    if _fasttext_model is not None:
        return _fasttext_model
    
    try:
        import fasttext
        # Try common paths
        for path in [
            'lid.176.bin',
            './lid.176.bin',
            '../lid.176.bin',
            'models/lid.176.bin',
        ]:
            import os
            if os.path.exists(path):
                _fasttext_model = fasttext.load_model(path)
                FASTTEXT_MODEL_PATH = path
                return _fasttext_model
    except Exception:
        pass
    
    return None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class LanguageResult:
    """
    Language detection result with confidence scoring.
    
    Attributes:
        language: Detected language ("vi", "en", or "unknown")
        confidence: Confidence score (0.0 to 1.0)
        is_uncertain: True if confidence is below threshold
    """
    language: str
    confidence: float
    is_uncertain: bool
    
    def __post_init__(self):
        if self.confidence < LOW_CONFIDENCE:
            object.__setattr__(self, 'is_uncertain', True)
        else:
            object.__setattr__(self, 'is_uncertain', False)


# =============================================================================
# STAGE 1: Diacritic Detection
# =============================================================================

def _has_vietnamese_diacritics(query: str) -> bool:
    """
    Stage 1: Fast check for Vietnamese diacritics.
    
    This is the FASTEST way to detect Vietnamese with HIGHEST confidence.
    If any Vietnamese diacritic is found, it's definitely Vietnamese.
    
    Why this works:
    - English has no diacritics
    - Vietnamese diacritics (àáạảã...) are unique to Vietnamese
    - This check is O(n) where n = query length, and just one regex search
    
    Args:
        query: User query string
        
    Returns:
        True if Vietnamese diacritics found, False otherwise
    """
    return bool(DIACRITIC_RE.search(query))


# =============================================================================
# STAGE 2: Token-Based Scoring
# =============================================================================

def _tokenize_query(query: str) -> list[str]:
    """
    Proper tokenization of query string.
    
    Uses regex instead of simple split() to properly handle:
    - Vietnamese characters with diacritics
    - Contractions (don't, I'm, etc.)
    - Technical terms
    
    Args:
        query: User query string
        
    Returns:
        List of tokens (lowercase)
    """
    return [token.lower() for token in TOKEN_RE.findall(query)]


def _calculate_english_score(tokens: list[str]) -> float:
    """
    Calculate English language score based on token intersection.
    
    Uses set intersection for O(n) performance instead of repeated checks.
    
    Why this approach:
    - Single set intersection is much faster than multiple regex searches
    - Strong indicators can override weak Vietnamese signals
    - Tokens are pre-computed, not recomputed
    
    Args:
        tokens: List of query tokens (lowercase)
        
    Returns:
        English score (0.0 to 1.0)
    """
    if not tokens:
        return 0.0
    
    token_set = frozenset(tokens)
    
    # Strong English indicators (high weight)
    strong_matches = len(token_set & ENGLISH_STRONG)
    
    # Medium English indicators (lower weight)
    medium_matches = len(token_set & ENGLISH_MEDIUM)
    
    # Calculate weighted score
    # Strong matches count more heavily
    weighted_score = (strong_matches * 2 + medium_matches) / len(tokens)
    
    return min(weighted_score, 1.0)


def _calculate_vietnamese_score(tokens: list[str]) -> float:
    """
    Calculate Vietnamese language score using pattern matching and word sets.
    
    This is more sophisticated than English scoring because Vietnamese
    without diacritics requires pattern recognition, not just word matching.
    
    Why patterns are necessary:
    - Users often type Vietnamese WITHOUT diacritics
    - Common typos like "la gif" instead of "là gì"
    - Single words like "co", "con" are ambiguous
    
    Args:
        tokens: List of query tokens (lowercase)
        
    Returns:
        Vietnamese score (0.0 to 1.0)
    """
    if not tokens:
        return 0.0
    
    # Rebuild lowercase string for pattern matching
    query_lower = ' '.join(tokens)
    token_set = frozenset(tokens)
    
    # Check patterns first (high weight)
    pattern_score = 0.0
    pattern_weight = 0.0
    
    for pattern, weight in VIETNAMESE_PATTERNS:
        if pattern.search(query_lower):
            pattern_score += weight
            pattern_weight += 1.0
    
    # Check word set matches
    word_matches = len(token_set & VIETNAMESE_STRONG)
    word_score = word_matches / len(tokens)
    
    # Combine scores - patterns are MUCH stronger indicators than words
    # When patterns match, they should dominate the score
    if pattern_weight > 0:
        # 80% patterns, 20% words - patterns are decisive
        pattern_avg = pattern_score / pattern_weight
        combined = (pattern_avg * 0.8) + (word_score * 0.2)
    else:
        combined = word_score
    
    return min(combined, 1.0)


def _calculate_confidence(en_score: float, vi_score: float) -> tuple[str, float]:
    """
    Determine language with confidence based on score difference.
    
    Why this approach:
    - Not just comparing raw scores
    - Considers the GAP between scores
    - High gap = high confidence
    
    Args:
        en_score: English score (0.0 to 1.0)
        vi_score: Vietnamese score (0.0 to 1.0)
        
    Returns:
        Tuple of (language, confidence)
    """
    score_diff = abs(en_score - vi_score)
    
    if en_score > vi_score:
        # English is stronger, but by how much?
        if score_diff > 0.4:
            return "en", 0.95
        elif score_diff > 0.2:
            return "en", 0.7
        else:
            return "en", 0.5  # Uncertain
    elif vi_score > en_score:
        # Vietnamese is stronger
        if score_diff > 0.4:
            return "vi", 0.95
        elif score_diff > 0.2:
            return "vi", 0.7
        else:
            return "vi", 0.5  # Uncertain
    else:
        # Equal scores - very uncertain
        return "unknown", 0.3


# =============================================================================
# STAGE 3: FastText Fallback
# =============================================================================

def _fasttext_detect(query: str) -> str:
    """
    FastText language detection (only for uncertain cases).
    
    IMPORTANT: This is ONLY called when confidence is LOW.
    If Vietnamese diacritics exist, this is never called.
    
    Why this approach:
    - FastText model loading is expensive
    - Most queries are clear (diacritics or strong keywords)
    - Only use FastText as fallback for truly ambiguous cases
    
    Args:
        query: User query string
        
    Returns:
        Language code ("vi" or "en")
    """
    model = _load_fasttext_model()
    
    if model is None:
        # FastText not available - default to English for safety
        return "en"
    
    # Predict - FastText returns like "__label__en" or "__label__vi"
    prediction = model.predict(query.replace('\n', ' '), k=1)
    label = prediction[0][0]
    
    if 'vi' in label:
        return "vi"
    elif 'en' in label:
        return "en"
    else:
        return "unknown"


# =============================================================================
# MAIN: Hybrid Detection Pipeline
# =============================================================================

def detect_language(query: str) -> LanguageResult:
    """
    Main language detection function - hybrid three-stage pipeline.
    
    Architecture:
    1. Diacritic check (fast, high confidence) - returns immediately if found
    2. Token scoring (fast, good confidence) - returns if clear winner
    3. FastText fallback (slower, for uncertain cases only)
    
    Args:
        query: User query string
        
    Returns:
        LanguageResult with language, confidence, and uncertainty flag
    """
    if not query or not query.strip():
        return LanguageResult(language="unknown", confidence=0.0, is_uncertain=True)
    
    # Stage 1: Diacritic detection (FAST - single regex check)
    if _has_vietnamese_diacritics(query):
        return LanguageResult(language="vi", confidence=0.98, is_uncertain=False)
    
    # Stage 2: Token-based scoring (FAST - single tokenization + set ops)
    tokens = _tokenize_query(query)
    
    if not tokens:
        # No valid tokens found
        return LanguageResult(language="unknown", confidence=0.0, is_uncertain=True)
    
    en_score = _calculate_english_score(tokens)
    vi_score = _calculate_vietnamese_score(tokens)
    
    language, confidence = _calculate_confidence(en_score, vi_score)
    
    # If confidence is high enough, return immediately (no FastText needed)
    if confidence >= HIGH_CONFIDENCE:
        return LanguageResult(language=language, confidence=confidence, is_uncertain=False)
    
    # Stage 3: FastText fallback (only for uncertain cases with NO pattern matches)
    # Check if any Vietnamese patterns matched - if so, trust the heuristic
    query_lower = ' '.join(tokens)
    has_vietnamese_pattern = any(
        pattern.search(query_lower) for pattern, _ in VIETNAMESE_PATTERNS
    )
    
    # Only use FastText if no patterns matched and confidence is uncertain
    if not has_vietnamese_pattern:
        fasttext_lang = _fasttext_detect(query)
        if fasttext_lang != "unknown":
            return LanguageResult(language=fasttext_lang, confidence=0.6, is_uncertain=True)
    
    # Use heuristic result (patterns are strong indicator even with low confidence)
    return LanguageResult(language=language, confidence=confidence, is_uncertain=True)


# =============================================================================
# BACKWARD COMPATIBILITY: Legacy functions for existing code
# =============================================================================

def _detect_lang(query: str) -> str:
    """
    Legacy function for backward compatibility.
    
    Returns simple language code string instead of LanguageResult.
    
    Args:
        query: User query string
        
    Returns:
        Language code: "vi", "en", or "unknown"
    """
    result = detect_language(query)
    return result.language


def _detect_and_correct_language(query: str) -> tuple[str, str, str]:
    """
    Legacy function for backward compatibility.
    
    Returns tuple: (corrected_query, detected_lang, original_lang)
    Note: Current implementation focuses on detection, not correction.
    Typo correction would require additional NLP pipeline.
    
    Args:
        query: User query string
        
    Returns:
        Tuple of (corrected_query, detected_lang, original_lang)
    """
    result = detect_language(query)
    # For now, return query as-is (correction not implemented)
    # Could be extended with NLP correction pipeline
    return query, result.language, result.language


# =============================================================================
# Utility functions for testing/debugging
# =============================================================================

def get_detection_details(query: str) -> dict:
    """
    Get detailed detection information for debugging.
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with detection details
    """
    has_diacritics = _has_vietnamese_diacritics(query)
    tokens = _tokenize_query(query)
    en_score = _calculate_english_score(tokens)
    vi_score = _calculate_vietnamese_score(tokens)
    language, confidence = _calculate_confidence(en_score, vi_score)
    result = detect_language(query)
    
    return {
        "query": query,
        "has_diacritics": has_diacritics,
        "tokens": tokens,
        "en_score": en_score,
        "vi_score": vi_score,
        "heuristic_result": language,
        "heuristic_confidence": confidence,
        "final_result": result.language,
        "final_confidence": result.confidence,
        "is_uncertain": result.is_uncertain,
    }


# =============================================================================
# Example test cases for validation
# =============================================================================

if __name__ == "__main__":
    # Test Vietnamese cases (should all be 'vi')
    vietnamese_cases = [
        "transformer laf gifs",
        "attention mechanism la gif",
        "co the giai thcih transformer k",
        "tai s transformer hoat dogn",
        "attention mechanism hoat dong nhu nao",
        "transformer la gi",
        "cach attention hoat dong",
        "em la ai",
    ]
    
    # Test English cases (should all be 'en')
    english_cases = [
        "What is transformer?",
        "how does BERT work?",
        "attention mechanism function",
        "explain difference between encoder and decoder",
        "BERT vs GPT",
    ]
    
    print("=" * 70)
    print("Language Detection - Vietnamese Test Cases")
    print("=" * 70)
    
    for query in vietnamese_cases:
        result = detect_language(query)
        status = "PASS" if result.language == "vi" else "FAIL"
        print(f"{status} {query} -> {result.language} (conf: {result.confidence:.2f})")
    
    print("\n" + "=" * 70)
    print("Language Detection - English Test Cases")
    print("=" * 70)
    
    for query in english_cases:
        result = detect_language(query)
        status = "PASS" if result.language == "en" else "FAIL"
        print(f"{status} {query} -> {result.language} (conf: {result.confidence:.2f})")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    vi_pass = sum(1 for q in vietnamese_cases if detect_language(q).language == "vi")
    en_pass = sum(1 for q in english_cases if detect_language(q).language == "en")
    
    print(f"Vietnamese: {vi_pass}/{len(vietnamese_cases)} passed")
    print(f"English: {en_pass}/{len(english_cases)} passed")