"""CoT-free native agent conversion; tool arguments are preserved exactly."""
import copy
import json
import re

MEMORY_MARKERS=('<|memory_start|>','<|memory_end|>','<|memory|>')
TERMINAL_TOOL={'type':'function','function':{
    'name':'terminal_step',
    'description':'Send an ordered batch of keystrokes to the persistent Linux terminal. Each duration is the number of seconds to wait after that command. Use an empty batch to wait or confirm completion. Set task_complete when the task is complete.',
    'parameters':{'type':'object','properties':{
        'commands':{'type':'array','items':{'type':'object','properties':{
            'keystrokes':{'type':'string'},'duration':{'type':'number'}},'required':['keystrokes']}},
        'task_complete':{'type':'boolean'}},'required':['commands','task_complete']}}}


def strip_cot(text):
    if text is None:return ''
    if not isinstance(text,str):raise ValueError('unsupported assistant content type')
    # Strip only a leading reasoning annotation, never text inside a command.
    while True:
        match=re.match(r'\s*<(think|analysis)>',text)
        if not match:break
        end=text.find('</'+match[1]+'>',match.end())
        if end<0:raise ValueError('unclosed leading reasoning tag')
        text=text[end+len(match[1])+3:]
    return text


def clean_native(row, source):
    messages=copy.deepcopy(row.get('messages') or row.get('conversations'))
    if isinstance(messages,str):messages=json.loads(messages)
    if not isinstance(messages,list) or not messages:raise ValueError('missing messages')
    messages=[json.loads(m) if isinstance(m,str) else m for m in messages]
    for message in messages:
        for key in ('reasoning_content','reasoning','analysis','thinking'):
            message.pop(key,None)
        if message.get('role')=='assistant':
            message['content']=strip_cot(message.get('content'))
            if message.get('function_call'):raise ValueError('legacy function_call requires explicit conversion')
            message.pop('function_call',None)
    result={'schema_version':2,'data_type':'agent_trajectory','source_dataset':source,
        'sub_dataset':row.get('trace_source') or 'native_agent','compression_scope':'none',
        'messages':messages,'tools':copy.deepcopy(row.get('tools')),
        'source_row_id':str(row.get('uuid') or row.get('run_id') or '')}
    result=normalize_qwen_tools(result)
    validate_clean(result)
    return result


def normalize_qwen_tools(row):
    """Decode transport JSON once; keep argument values and tool results intact.

    Qwen's shipped template supplies <tools>, <tool_call>, <tool_response>.
    Never manually embed a second tool protocol in message content.
    """
    row=copy.deepcopy(row)
    tools=row.get('tools') or []
    if isinstance(tools,str):tools=json.loads(tools)
    if not isinstance(tools,list):raise ValueError('tools must be a list')
    names=set()
    definitions={}
    normalized_tools=[]
    for i,tool in enumerate(tools):
        if isinstance(tool,str):tool=json.loads(tool);tools[i]=tool
        if tool.get('type')!='function':raise ValueError('unsupported tool type')
        fn=tool.get('function')
        if not isinstance(fn,dict) or not isinstance(fn.get('name'),str):raise ValueError('invalid tool definition')
        params=fn.get('parameters',{})
        if isinstance(params,str):params=json.loads(params)
        if not isinstance(params,dict):raise ValueError('invalid tool parameters')
        fn['parameters']=params
        if fn['name'] in definitions:
            if tool!=definitions[fn['name']]:raise ValueError('conflicting tool definitions')
            continue
        names.add(fn['name'])
        definitions[fn['name']]=tool
        normalized_tools.append(tool)
    pending={}
    for index,message in enumerate(row['messages']):
        role=message.get('role')
        if role not in ('system','user','assistant','tool'):raise ValueError('unsupported message role')
        if message.get('content') is None:message['content']=''
        if role=='tool' and not isinstance(message['content'],str):
            # v1 observations can be JSON objects/lists. Qwen's template would
            # silently render non-string content as empty without this step.
            message['content']=json.dumps(message['content'],ensure_ascii=False,allow_nan=False)
        if not isinstance(message['content'],str):raise ValueError('non-text message content')
        if role=='tool':
            call_id=message.get('tool_call_id')
            if not call_id:
                matching=[i for i,name in pending.items() if not message.get('name') or message['name']==name]
                if len(matching)==1:
                    call_id=matching[0]
                    message['tool_call_id']=call_id
                else:raise ValueError('ambiguous tool response without id')
            if call_id not in pending:raise ValueError('unmatched tool response')
            name=pending.pop(call_id)
            if message.get('name') and message['name']!=name:raise ValueError('tool response name mismatch')
            message['name']=name
        elif pending:raise ValueError('missing tool response before next turn')
        calls=message.get('tool_calls') or []
        if isinstance(calls,str):calls=json.loads(calls)
        if calls and role!='assistant':raise ValueError('tool calls outside assistant turn')
        for call in calls:
            fn=call.get('function',{})
            if call.get('type')!='function' or fn.get('name') not in names:raise ValueError('undefined tool call')
            args=fn.get('arguments')
            if isinstance(args,str):args=json.loads(args)
            if not isinstance(args,dict):raise ValueError('tool arguments must be an object')
            fn['arguments']=args
            call.pop('index',None)
            call_id=call.get('id')
            if not isinstance(call_id,str) or not call_id or call_id in pending:raise ValueError('invalid tool call id')
            pending[call_id]=fn['name']
        if calls:message['tool_calls']=calls
        else:message.pop('tool_calls',None)
    # A final assistant call can be a valid next-action SFT target. Any earlier
    # unresolved call has already failed above; no fabricated observation.
    row['tools']=normalized_tools
    return row


def clean_nemotron(row,source,subset):
    result=clean_native(row,source)
    result['sub_dataset']=subset
    if source=='nvidia/Nemotron-SFT-Agentic-v2' and subset=='search':
        for message in result['messages']:
            if message['role']!='system':continue
            text=message['content']
            # This source's generation harness requests visible Thought/Search/
            # Observation prose even though it stores native calls separately.
            # Remove that harness-only format, not the task or answer contract.
            text=re.sub(r'(?m)^\d+\. \*\* Planning \*\*[^\n]*\n?', '', text)
            text=text.replace('Keep a counter in your thought process to track the number of tool calls you have made. ', '')
            text=re.sub(r'(?s)\n\d+\. \*\* Response Format \*\*.*$', '', text)
            message['content']=text.rstrip()+'\nUse native tool calls. Do not expose hidden reasoning.'
    return result


def clean_openthoughts(row, source='open-thoughts/OpenThoughts-Agent-SFT-100K'):
    original=row.get('conversations') or row.get('messages')
    if not original:raise ValueError('missing conversations')
    if any(m.get('tool_calls') for m in original):return clean_native(row,source)
    first=original[0].get('content','')
    if 'Task Description:' not in first:raise ValueError('unknown terminal harness prompt')
    messages=[{'role':'system','content':
        'Solve the user task in the persistent Linux terminal using terminal_step. '
        'Commands are sent verbatim; include a newline to execute them. '
        'Special key sequences use tmux notation (C-c, C-d). '
        'Use duration to wait for output. Mark task_complete when finished.'},
        {'role':'user','content':first.split('Task Description:',1)[1].lstrip()}]
    pending=None
    for index,message in enumerate(original[1:],start=1):
        if message['role']=='assistant':
            if pending is not None:raise ValueError('missing terminal response')
            text=strip_cot(message.get('content')).strip()
            if text.startswith('```'):
                text=re.sub(r'^```(?:json)?\s*','',text)
                text=re.sub(r'\s*```$','',text)
            action=json.loads(text)
            commands=action.get('commands')
            if not isinstance(commands,list):raise ValueError('missing command batch')
            for command in commands:
                if not isinstance(command,dict) or not isinstance(command.get('keystrokes'),str):
                    raise ValueError('invalid terminal command')
            pending=f'terminal_{index}'
            messages.append({'role':'assistant','content':'','tool_calls':[{
                'id':pending,'type':'function','function':{'name':'terminal_step',
                'arguments':{'commands':commands,'task_complete':bool(action.get('task_complete',False))}}}]})
        elif message['role']=='user' and pending is not None:
            messages.append({'role':'tool','tool_call_id':pending,'name':'terminal_step',
                             'content':message['content']})
            pending=None
        else:raise ValueError('unexpected terminal role sequence')
    result={'schema_version':2,'data_type':'agent_trajectory','source_dataset':source,
        'sub_dataset':row.get('trace_source') or 'terminal','compression_scope':'none',
        'messages':messages,'tools':[copy.deepcopy(TERMINAL_TOOL)],
        'source_row_id':str(row.get('run_id') or row.get('trial_name') or '')}
    validate_clean(result)
    return result


def validate_clean(row):
    serialized=json.dumps({'messages':row['messages'],'tools':row['tools']},ensure_ascii=False)
    if any(marker in serialized for marker in MEMORY_MARKERS):raise ValueError('memory marker in plain agent row')
    for message in row['messages']:
        if any(k in message for k in ('reasoning_content','reasoning','analysis','thinking')):
            raise ValueError('reasoning field remains')
        if message.get('role')=='assistant' and re.search(r'</?(think|analysis)>',message.get('content') or ''):
            raise ValueError('unparsed reasoning marker')
