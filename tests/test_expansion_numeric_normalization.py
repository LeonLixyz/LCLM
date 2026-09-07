from data.real_expansion_agent import _normalized_tokens


def test_numeric_sign_and_unit_cannot_disappear_in_exact_match():
    assert _normalized_tokens('-23%') != _normalized_tokens('23%')
    assert _normalized_tokens('23%') != _normalized_tokens('23')
    assert _normalized_tokens('1.23') != _normalized_tokens('1,23')


def test_text_matching_still_ignores_case_and_sentence_punctuation():
    assert _normalized_tokens('All Cash.') == _normalized_tokens('all cash')
