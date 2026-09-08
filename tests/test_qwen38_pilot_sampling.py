import pytest
from data.qwen38_pilot_sampling import sampling_kwargs, output_name


def test_matched_preserves_original_request():
    assert sampling_kwargs('matched') == {
        'temperature': 0, 'max_tokens': 2048,
        'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}


def test_recommended_keeps_nonthinking_and_budget():
    assert sampling_kwargs('recommended') == {
        'temperature': 0.7, 'top_p': 0.8, 'presence_penalty': 1.5,
        'max_tokens': 2048, 'extra_body': {'top_k': 20, 'min_p': 0.0,
        'repetition_penalty': 1.0, 'chat_template_kwargs': {'enable_thinking': False}}}
    assert output_name('recommended') != output_name('matched')


@pytest.mark.parametrize('profile', ['matched', 'recommended'])
def test_kwargs_not_shared_across_requests(profile):
    first = sampling_kwargs(profile)
    first['extra_body']['chat_template_kwargs']['enable_thinking'] = True
    assert sampling_kwargs(profile)['extra_body']['chat_template_kwargs']['enable_thinking'] is False


@pytest.mark.parametrize('profile', ['', '../v1', 'full', None])
def test_unknown_profiles_rejected(profile):
    with pytest.raises(ValueError):
        sampling_kwargs(profile)
    with pytest.raises(ValueError):
        output_name(profile)
