"""Identify punctuation-normalization collisions; never automatically reject rows."""
import re


def accounting_tokens(text):
    # Diagnostic only: parentheses can also be annotations, not negative values.
    # Surrounding decrease/change wording requires evidence review.
    text = re.sub(r"\(\s*[$€£]?\s*(\d+(?:[.,]\d+)*%?)\s*\)", r"-\1", text)
    return re.findall(r"[-+]?\d+(?:[.,]\d+)*%?|[a-z]+", text.casefold())


def accounting_collision(candidate, reference):
    classic = lambda text: re.findall(r"[-+]?\d+(?:[.,]\d+)*%?|[a-z]+", text.casefold())
    return (classic(candidate) == classic(reference)
            and accounting_tokens(candidate) != accounting_tokens(reference))
