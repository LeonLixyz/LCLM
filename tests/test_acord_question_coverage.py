import pytest
from data.acord_question_coverage import parse_question, PREFIX, FINAL, SCALES


def question(query='as is', scale='beir_0_4'):
    return 'Use these source documents: abc.\n'+PREFIX+query+SCALES[scale]+FINAL


@pytest.mark.parametrize('query', ['as is', 'multiple\nquery lines', 'period. and "quotes"'])
def test_preserves_exact_query(query):
    assert parse_question(question(query)) == (query, 'beir_0_4')


def test_prepared_legacy_is_not_accepted_as_saved_normalized_question():
    raw = question(scale='legacy_prepared_1_5')
    with pytest.raises(ValueError): parse_question(raw)
    assert parse_question(raw, allow_legacy=True) == ('as is', 'legacy_prepared_1_5')


@pytest.mark.parametrize('raw', [question(''), question().replace(FINAL, ''),
    question().replace('0 (irrelevant)', '9 (irrelevant)'), question().replace('Use these source documents:', 'Wrong:')])
def test_malformed_question_fails(raw):
    with pytest.raises(ValueError): parse_question(raw, allow_legacy=True)
