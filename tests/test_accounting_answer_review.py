import pytest
from data.accounting_answer_review import accounting_collision, accounting_tokens


@pytest.mark.parametrize('candidate,reference', [('$(3)', '3'), ('($3)', '3'), ('(3.5%)', '3.5%'), ('3', '(3)')])
def test_flags_potential_collision(candidate, reference):
    assert accounting_collision(candidate, reference)


@pytest.mark.parametrize('candidate,reference', [('3', '3'), ('-3', '3'), ('(3)', '(3)'), ('(note)', 'note'), ('3 million', '3 billion')])
def test_no_new_collision(candidate, reference):
    assert not accounting_collision(candidate, reference)


def test_currency_commas_and_percent_are_preserved():
    assert accounting_tokens('($1,234.50) (2.5%)') == ['-1,234.50', '-2.5%']
