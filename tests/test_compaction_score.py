import pytest

from compaction.math_answers import correct, last_boxed


def test_nested_final_box():
    assert last_boxed(r'\boxed{2} then \boxed{\frac{3}{4}}') == r'\frac{3}{4}'


def test_incomplete_final_box_fails():
    assert last_boxed(r'\boxed{2} then \boxed{') is None
    assert last_boxed('The answer is 2.') is None


def test_equivalent_math():
    pytest.importorskip('math_verify')
    assert correct('1/2', r'\frac{2}{4}')
    assert not correct('2', '3')


def test_missing_prediction_fails():
    pytest.importorskip('math_verify')
    assert not correct('2', None)
