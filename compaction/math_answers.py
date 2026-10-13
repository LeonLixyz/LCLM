"""MATH answer extraction and symbolic scoring (math-verify 0.9.0)."""
import re


def last_boxed(text):
    """Contents of the last balanced ``\\boxed{...}``, or ``None``."""
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        slashes = 0
        j = i - 1
        while j >= 0 and text[j] == '\\':
            slashes += 1
            j -= 1
        if slashes % 2:
            continue
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        if depth == 0:
            return text[start:i]
    return None


def correct(answer, prediction):
    from math_verify import LatexExtractionConfig, parse, verify
    cfg = [LatexExtractionConfig(boxed_match_priority=0)]
    gold = parse('\\boxed{' + answer + '}', extraction_config=cfg)
    assert gold, 'Reference answer must be parseable'
    pred = parse('\\boxed{' + prediction + '}', extraction_config=cfg) if prediction is not None else []
    return bool(verify(gold, pred))
