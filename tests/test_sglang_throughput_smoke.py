import pytest
from data.sglang_throughput_smoke import first_request, inspect_response, graph_capture_lines, report_requests


def test_first_request_requires_captured_non_thinking_initial_tool_turn():
    request = {'messages': [{}, {}], 'tools': [{}],
               'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
    assert first_request([{'phase': 'rollout', 'request': request, 'response': {}}]) == request
    with pytest.raises(ValueError):
        first_request([{'phase': 'judge', 'request': request, 'response': {}}])
    with pytest.raises(ValueError):
        first_request([{'phase': 'rollout', 'request': request}])


def response(segment='seg_1'):
    return {'choices': [{'response_token_ids': [42], 'finish_reason': 'tool_calls',
        'message': {'content': '', 'tool_calls': [{'id': 'call1', 'type': 'function',
            'function': {'name': 'expand', 'arguments': '{"segment_id":"' + segment + '"}'}}]}}]}


def test_parser_check_requires_known_segment_and_complete_response():
    assert inspect_response(response(), {'seg_1'})['valid_native_call']
    with pytest.raises(ValueError):
        inspect_response(response('seg_2'), {'seg_1'})
    broken = response()
    broken['choices'][0]['finish_reason'] = 'length'
    with pytest.raises(RuntimeError):
        inspect_response(broken, {'seg_1'})


def test_capture_evidence_excludes_disable_message():
    log = 'Disable cuda graph\nCapture cuda graph begin\nCapture cuda graph end\nReady'
    assert len(graph_capture_lines(log)) == 2


def test_rate_is_requests_not_complete_rollouts():
    result = report_requests([{'seconds': 1, 'error': {'type': 'test'}}], 2)
    assert result['requests_per_second'] == .5
    assert result['counts']['requests'] == 1
