"""Training-split adapters and full-document segment construction."""
import hashlib
import json
import random
import re
from pathlib import Path

SOURCE_ROOT=Path('/data/stage3-agent/real-expansion/sources')

def load(key,path='materialized/default/train'):
    from datasets import load_from_disk
    return load_from_disk(str(SOURCE_ROOT/key/path))

def doc(identifier,text,title=''):
    return {'document_id':str(identifier),'text':str(text),'title':str(title or identifier)}

def answer_text(value):
    if isinstance(value,list):return ' | '.join(map(str,value))
    return str(value)


def word_chunks(text,size=768):
    """Bound word count without destroying tables, newlines, or source spacing."""
    words=list(re.finditer(r'\S+',text))
    for i in range(0,len(words),size):
        start=0 if i==0 else words[i].start()
        end=words[i+size].start() if i+size<len(words) else len(text)
        yield text[start:end]

def financial_documents(key):
    result={}
    for r in load(key,'materialized_raw/documents/train'):
        table=json.loads(r['table_original_json']) or json.loads(r['table_json'])
        if isinstance(table,dict):table=table.get('table',[])
        text='\n'.join(json.loads(r['pre_text_json']))+'\n'+ '\n'.join(' | '.join(map(str,t)) for t in table)+'\n'+'\n'.join(json.loads(r['post_text_json']))
        result[r['document_id']]=doc(r['document_id'],text)
    return result

def candidates(key):
    """Yield question, answer, original documents; never load an evaluation split."""
    if key in ('maud','finqa','pubmedqa_labeled','clapnq','contract_nli'):
        from data.build_real_expansion_pilot_modal import _maud,_finqa,_pubmedqa,_clapnq,_contract_nli
        fn={'maud':_maud,'finqa':_finqa,'pubmedqa_labeled':_pubmedqa,'clapnq':_clapnq,'contract_nli':_contract_nli}[key]
        documents,rows=fn(); byid={r['document_id']:r for r in documents}
        for r in rows:yield r['source_row_id'],r['question'],r['answer'],[byid[r['document_id']]]
    elif key in ('tatqa','convfinqa'):
        documents=financial_documents(key)
        for r in load(key,'materialized_raw/tasks/train'):
            value=json.loads(r['answer_json']) if key=='tatqa' else r['answer']
            yield r['task_id'],r['question'],answer_text(value),[documents[r['document_id']]]
    elif key=='multihiertt':
        for r in load(key):
            text='\n\n'.join(r['paragraphs']+r['tables'])
            yield r['uid'],r['question'],r['answer'],[doc(r['uid'],text)]
    elif key=='multidoc2dial':
        for r in load(key,'materialized/multidoc2dial/train'):
            yield r['id'],r['question'].replace('[SEP]','\n'),r['utterance'],[doc(r['title'],r['context'],r['title'])]
    elif key=='faithdial':
        for i,r in enumerate(load(key,'materialized/plain_text/train')):
            yield str(i),'Respond to the last user turn using the source:\n'+'\n'.join(r['history']),r['response'],[doc(i,r['knowledge'])]
    elif key=='watsonx_docs_qa':
        documents={r['doc_id']:doc(r['doc_id'],r['document'],r['title']) for r in load(key,'materialized/corpus/train')}
        for r in load(key,'materialized/question_answers/train'):
            ids=re.findall(r'[A-Fa-f0-9]{40}',r['correct_answer_document_ids'])
            if ids and all(i in documents for i in ids):
                yield r['question_id'],r['question'],r['correct_answer'],[documents[i] for i in ids]
    elif key=='cuad':
        documents={r['document_id']:doc(r['document_id'],r['text'],r['title']) for r in load(key,'materialized_raw/documents/train')}
        for r in load(key,'materialized_raw/tasks/train'):
            if r['is_impossible']:continue
            answers=[x['text'] for x in json.loads(r['answers_json'])]
            if answers:yield r['task_id'],r['question'],answer_text(answers),[documents[r['document_id']]]
    elif key=='acord':
        documents={r['document_id']:doc(r['document_id'],r['text']) for r in load(key,'materialized_raw/documents/train')}
        for r in load(key,'materialized_raw/tasks/train'):
            yield r['task_id'],f'Rate how relevant the cited contract clause is to this request: {r["query"]}. Return the integer relevance grade from 1 (irrelevant) to 5 (highly relevant).',str(r['relevance']),[documents[r['document_id']]]
    elif key=='billsum':
        for i,r in enumerate(load(key)):
            yield str(i),'Summarize the supplied bill.',r['summary'],[doc(i,r['text'],r['title'])]
    elif key=='lex_glue':
        for config in ['case_hold','ecthr_a','ecthr_b','eurlex','ledgar','scotus','unfair_tos']:
            ds=load(key,f'materialized/{config}/train')
            feature=ds.features.get('label') or ds.features.get('labels')
            if hasattr(feature,'feature'):feature=feature.feature
            names=getattr(feature,'names',None)
            for i,r in enumerate(ds):
                if config=='case_hold':
                    text=r['context']; choices=r['endings']; answer=str(r['label'])
                    question='Which holding best completes the case? Return its zero-based index.\n'+'\n'.join(f'{j}: {x}' for j,x in enumerate(choices))
                else:
                    text=r['text'];text='\n'.join(text) if isinstance(text,list) else text
                    labels=r.get('label',r.get('labels')); labels=labels if isinstance(labels,list) else [labels]
                    if not names:continue
                    answer=' | '.join(names[n] for n in sorted(labels)) or 'none'
                    question=f'Classify this document for {config}. Choose labels from: '+', '.join(names)+'. Return labels in the listed order, separated by |; use none when no label applies.'
                yield f'{config}-{i}',question,answer,[doc(f'{config}-{i}',text)]


def build_task(key,source_id,identifier,question,answer,documents,pool,seed=20260906):
    from data.synthetic_expansion_agent import format_training_user_prompt,format_rollout_user_prompt
    rng=random.Random(f'{seed}:{key}:{identifier}')
    if not question.strip() or not answer.strip():raise ValueError('empty_qa')
    support_ids={d['document_id'] for d in documents}
    companions=[d for d in pool if d['document_id'] not in support_ids]
    rng.shuffle(companions)
    segments=[]
    for d in documents:
        words=d['text'].split()
        if not words:raise ValueError('empty_document')
        # Keep the complete source, including all tables and evidence.
        pieces=list(word_chunks(d['text']))
        for part,text in enumerate(pieces):
            words=text.split()
            if len(words)<512:
                for companion in companions:
                    text+='\nRELATED SOURCE '+companion['document_id']+':\n'+companion['text']
                    if len(text.split())>=512:break
                text=next(word_chunks(text))
            if len(text.split())<512:raise ValueError('insufficient_source_text')
            segments.append({'record_id':d['document_id']+f':part{part}','text':text,
                'summary':d['document_id']+' — '+d['title']+f' (part {part+1}): '+' '.join(words[:32]),'_support':True})
    if len(segments)>12:raise ValueError('source_exceeds_12_segments')
    distractor_buffer=[]
    for d in companions:
        if len(segments)>=max(8,len([s for s in segments if s['_support']])+2):break
        distractor_buffer.append(d)
        text='\n\n'.join('SOURCE '+x['document_id']+' — '+x['title']+'\n'+x['text'] for x in distractor_buffer)
        if len(text.split())<512:continue
        text=next(word_chunks(text))
        segments.append({'record_id':'+'.join(x['document_id'] for x in distractor_buffer),'text':text,
            'summary':'; '.join(x['document_id']+' — '+x['title'] for x in distractor_buffer)[:400]+': '+' '.join(text.split()[:32]),'_support':False})
        distractor_buffer=[]
    if len(segments)<2:raise ValueError('insufficient_segments')
    rng.shuffle(segments)
    support=[]
    for i,s in enumerate(segments,1):
        s['segment_id']=f'seg_{i}'
        if s.pop('_support'):support.append(s['segment_id'])
    # Distractor contracts can answer the same generic clause question. State
    # which source is in scope; this is document identity, not an answer hint.
    question='Use these source documents: '+', '.join(d['document_id'] for d in documents)+'.\n'+question
    question+='\nReturn exactly FINAL: followed by your answer.'
    task_id='rea3-'+hashlib.sha256(f'{seed}:{key}:{identifier}'.encode()).hexdigest()[:24]
    task={'schema_version':2,'generator_version':'full-real-expansion-v3','task_id':task_id,
        'source_dataset':source_id,'source_row_id':identifier,'family':key,
        'question':question,'raw_question':question,'gold_answer':answer,'expected_final':'FINAL: '+answer,
        'segments':segments,'support_segment_ids':support,'seed':seed}
    task['training_user_prompt']=format_training_user_prompt(segments,question)
    task['user_prompt']=task['training_user_prompt']
    task['rollout_user_prompt']=format_rollout_user_prompt(segments,question)
    return task
