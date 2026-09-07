"""Harvest native calls and one explicit final answer, without teacher prose."""
import copy
import re


def harvest_training_messages(messages):
    """Remove assistant preambles, never rewrite tool calls or observations.

    The requested task contract is one FINAL line. Reject missing/ambiguous
    boundaries and trailing text instead of guessing where the answer ends.
    Correctness must be checked again on the returned messages.
    """
    result=copy.deepcopy(messages)
    if not result or result[-1].get('role')!='assistant' or result[-1].get('tool_calls'):
        raise ValueError('Missing terminal assistant answer')
    final=result[-1].get('content')
    if not isinstance(final,str):raise ValueError('Final answer must be text')
    matches=list(re.finditer(r'^FINAL:[ \t]*(\S[^\r\n]*)',final,re.M))
    if len(matches)!=1:raise ValueError('Expected exactly one FINAL line')
    match=matches[0]
    if final[match.end():].strip():raise ValueError('Unexpected text after FINAL line')
    payload=match.group(1).strip()
    if re.search(r'</?(?:think|analysis)>|<\|(?:memory|im_start|im_end)',payload,re.I):
        raise ValueError('Unexpected control/reasoning marker in final answer')
    removed=0
    for message in result[:-1]:
        if message.get('role')!='assistant':continue
        if not message.get('tool_calls'):raise ValueError('Unexpected intermediate text-only assistant')
        removed+=len(message.get('content') or '')
        message['content']=''
    normalized='FINAL: '+payload
    removed+=max(0,len(final)-len(normalized))
    result[-1]['content']=normalized
    for message in result:
        if message.get('role')=='assistant':
            for key in ('reasoning','reasoning_content','analysis','thinking'):
                message.pop(key,None)
    return result,{'version':'native-calls-and-explicit-final-v1','removed_assistant_characters':removed}
