"""Independent format, evidence and label checks for saved expansion traces."""
import re
from data.synthetic_expansion_agent import EXPAND_TOOL, TEACHER_SYSTEM_PROMPT, verify_trace
from data.expansion_task_normalization import TASK_SYSTEM_PROMPT
from data.harvest_expansion_trace import harvest_training_messages
from data.real_expansion_agent import verify_real_trace


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_trace(row, source, tokenizer, prep, pubmed_training_ids=None):
    messages = row['messages']
    require(row['verification'].get('accepted') is True, 'Unaccepted row')
    require(row['tools'] == [EXPAND_TOOL], 'Wrong tool definition')
    require(messages[0] == {'role':'system','content':TASK_SYSTEM_PROMPT}, 'Wrong saved task system')
    require(messages[1]['role'] == 'user', 'Missing user context')
    require(all(TEACHER_SYSTEM_PROMPT not in (m.get('content') or '') for m in messages), 'Teacher prompt leak')
    require(harvest_training_messages(messages)[0] == messages, 'Assistant reasoning/preamble not harvested')
    matches = re.findall(r'(?m)^(seg_[1-9][0-9]*)\n<\|memory_start\|>(.*?)<\|memory_end\|>', messages[1]['content'], re.S)
    segments = dict(matches)
    require(len(matches) == len(segments) == row['segment_count'] and len(segments) >= 2, 'Invalid segment set')
    lengths = [len(tokenizer.encode(body, add_special_tokens=False)) for body in segments.values()]
    require(min(lengths) >= 512, 'Segment below 512 tokens')
    pending = {}; expanded = []; seen_calls = set(); calls = 0
    for message in messages[2:]:
        if message['role'] == 'assistant':
            for call in message.get('tool_calls', []):
                require(call['id'] not in seen_calls, 'Duplicate tool call ID')
                seen_calls.add(call['id'])
                require(call.get('type') == 'function' and call['function']['name'] == 'expand', 'Non-native expand call')
                arguments = call['function']['arguments']
                require(isinstance(arguments, dict) and set(arguments) == {'segment_id'}, 'Unexpected tool arguments')
                segment = arguments['segment_id']
                require(segment in segments, 'Unknown segment')
                pending[call['id']] = segment; calls += 1
        elif message['role'] == 'tool':
            require(message['tool_call_id'] in pending, 'Orphan or duplicate result')
            segment = pending.pop(message['tool_call_id'])
            require(message['content'] == segments[segment], 'Expanded text differs from original')
            expanded.append(segment)
        else:
            raise ValueError('Unexpected role after initial context')
    require(not pending and calls == row['tool_call_count'], 'Unanswered calls or incorrect call count')
    require(set(row['support_segment_ids']) <= set(expanded), 'Support not expanded')
    task = {'segments':[{'segment_id':k,'text':v} for k,v in matches],
        'support_segment_ids':row['support_segment_ids'], 'gold_answer':row['gold_answer'],
        'expected_final':'FINAL: '+str(row['gold_answer']), 'source_dataset':row['source_dataset']}
    if source == 'synthetic':
        require(verify_trace(task, messages).accepted, 'Synthetic answer verification failed')
    else:
        verdict = verify_real_trace(task, messages)
        if source in {'clapnq','multidoc2dial','faithdial','watsonx_docs_qa','billsum','pubmedqa_labeled'}:
            require(row['verification'].get('answer_metric') == 'qwen_question_evidence_dual_judge', 'Missing V6 semantic review')
            review = row['verification']['semantic_review']
            require(review.get('keep') is True and all(
                review[k].get('correct') is True and review[k].get('grounded') is True
                for k in ('blind_vote','reference_vote')), 'Failed semantic review')
            if source == 'billsum':
                claims = review['summary_claims']
                require(claims.get('keep') is True and bool(claims['claims']) and all(
                    v['supported'] is True and v['quotes_present'] is True for v in claims['claims']), 'Unsupported summary claims')
            if source == 'pubmedqa_labeled':
                require(pubmed_training_ids and row['source_row_id'] in pubmed_training_ids, 'Held-out biomedical task')
                for body in segments.values():
                    ids = set(re.findall(r'(?:RELATED )?SOURCE pubmed-(\d+)', body))
                    require(ids and ids <= pubmed_training_ids, 'Held-out biomedical document')
                require(verdict['accepted'], 'Wrong biomedical decision')
            else:
                require(verdict['reason'].startswith(('accepted:','wrong_answer:')), 'Invalid source structure')
        else:
            require(verdict['accepted'], 'Source answer verification failed')
    compact = prep.worker_process_example((row,16,'compression_prompt',None,None))
    require(compact is not None, 'Packing conversion failed')
    compact_messages, memories = prep._compact_agent_memory_regions(messages)
    require(memories == list(segments.values()) == compact['memory_strings'], 'Memory bodies changed in packing')
    rendered = tokenizer.apply_chat_template(compact_messages, tools=row['tools'], tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    spans = [(m.start(1),m.end()) for m in re.finditer(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>\n',rendered,re.S)]
    expected = [any(s <= a < b <= e for s,e in spans) for a,b in encoded['offset_mapping']]
    require(encoded['input_ids'] == compact['base_input_ids'], 'Packed tokens differ from template')
    require(expected == [v != -100 for v in compact['base_labels']], 'Assistant-only label mismatch')
    require(sum(rendered[a:b].count('<tool_call>\n') for a,b in spans) == calls, 'Not all calls are supervised')
    return {'accepted':1, 'calls':calls, 'multi_expansion':int(len(set(expanded))>1),
        'labeled_tokens':sum(expected), 'segments':len(segments), 'minimum_segment_tokens':min(lengths)}


def init_worker():
    from data import preprocess_for_dynamic_packing as prep
    from data.stage3_tokenizers import DECODER, DECODER_REVISION, ENCODER, ENCODER_REVISION
    prep.worker_init(DECODER, ENCODER, DECODER_REVISION, ENCODER_REVISION)


def audit_worker(payload):
    row, source, allowed = payload
    from data import preprocess_for_dynamic_packing as prep
    try:
        return {'task_id':row['task_id'], 'metrics':audit_trace(row,source,prep._worker_tokenizer,prep,allowed)}
    except Exception as exc:
        return {'task_id':row.get('task_id'), 'error':f'{type(exc).__name__}: {exc}'}
