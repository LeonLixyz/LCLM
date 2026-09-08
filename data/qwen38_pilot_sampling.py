"""Two bounded diagnostic configurations, never production defaults."""

FROZEN_INPUTS_SHA256 = 'ac6e54c31e7554c173ccd19513028d5d8d5d0b0096cfce78f8a4af70dac6f210'


def sampling_kwargs(profile):
    if profile not in ('matched', 'recommended'):
        raise ValueError('Unknown pilot sampling profile')
    result = {'temperature': 0, 'max_tokens': 2048,
              'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
    if profile == 'recommended':
        result.update(temperature=0.7, top_p=0.8, presence_penalty=1.5)
        result['extra_body'].update(top_k=20, min_p=0.0, repetition_penalty=1.0)
    return result


def output_name(profile):
    sampling_kwargs(profile)  # Reject unknown profiles before resolving a path.
    return ('qwen38-27b-teacher-pilot-v1' if profile == 'matched'
            else 'qwen38-27b-teacher-recommended-v2')
