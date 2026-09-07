"""Convert only TechQA training questions with exact, verified answer spans."""
from collections import Counter

def reference_documents(rows):
    if isinstance(rows, dict):
        rows = list(rows.values())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Missing TechQA held-out reference rows')
    result = set()
    for row in rows:
        if not isinstance(row, dict) or 'DOCUMENT' not in row:
            raise ValueError('Unknown TechQA held-out reference schema')
        value = row['DOCUMENT']
        if isinstance(value, str) and value not in ('-', ''):
            result.add(value)
    return result

def convert_training(training, documents, dev, validation):
    if not isinstance(training, list) or not isinstance(documents, dict):
        raise ValueError('Unknown TechQA question/document schema')
    heldout = reference_documents(dev) | reference_documents(validation)
    counts = Counter(); tasks = []; selected = {}; seen = set()
    for row in training:
        counts['input_questions'] += 1
        identifier = row.get('QUESTION_ID')
        if not isinstance(identifier, str) or not identifier.startswith('TRAIN_') or identifier in seen:
            raise ValueError('Non-training or duplicate TechQA question')
        seen.add(identifier)
        if row.get('ANSWERABLE') != 'Y':
            counts['unanswerable'] += 1
            continue
        doc_id = row.get('DOCUMENT')
        if doc_id in heldout:
            counts['heldout_document_overlap'] += 1
            continue
        document = documents.get(doc_id)
        if not isinstance(document, dict) or not isinstance(document.get('text'), str):
            counts['missing_document'] += 1
            continue
        text = document['text']; answer = row.get('ANSWER')
        try:
            start, end = int(row['START_OFFSET']), int(row['END_OFFSET'])
        except (KeyError, TypeError, ValueError):
            counts['invalid_offsets'] += 1
            continue
        if not isinstance(answer, str) or not answer or not 0 <= start < end <= len(text) or text[start:end] != answer:
            counts['answer_span_mismatch'] += 1
            continue
        title = row.get('QUESTION_TITLE') or ''
        body = row.get('QUESTION_TEXT') or row.get('QUESTION_BODY') or ''
        if not isinstance(title, str) or not isinstance(body, str) or not (title+body).strip():
            counts['empty_question'] += 1
            continue
        question = title.strip() if title.strip() == body.strip() else (title+'\n\n'+body).strip()
        tasks.append({'task_id':identifier,'question':question,'answer':answer,'document_id':doc_id,
                      'start_offset':start,'end_offset':end,'source_split':'training_Q_A.json'})
        selected[doc_id] = {'document_id':doc_id,'title':str(document.get('title') or doc_id),'text':text}
        counts['accepted'] += 1
    counts['documents'] = len(selected)
    return tasks, list(selected.values()), dict(counts), sorted(heldout)
