import json
import pytest
from data.prepare_teacher_transition import file_sha
from data.split_expansion_routes import split_source


def fixture(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    path=prepared/'s.tasks.jsonl'
    rows=[{'task_id':str(i),'_attempt_provenance':{'kind':kind}} for i,kind in
          enumerate(['retry_rejected','previously_unattempted','retry_rejected'])]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (prepared/'s.build.json').write_text(json.dumps({'prepared_files':{path.name:file_sha(path)},
        'counts':{'retry_rejected':2,'previously_unattempted':1}}))
    return prepared,tmp_path/'routes'


def test_exact_disjoint_routes_and_resume(tmp_path):
    prepared,out=fixture(tmp_path);report=split_source('s',prepared,out)
    assert [r['tasks'] for r in report['routes']]==[1,2]
    assert json.loads((out/'first235/s.tasks.jsonl').read_text())['task_id']=='1'
    assert [json.loads(x)['task_id'] for x in (out/'retry27/s.tasks.jsonl').read_text().splitlines()]==['0','2']
    assert split_source('s',prepared,out)==report


def test_changed_route_rejected(tmp_path):
    prepared,out=fixture(tmp_path);split_source('s',prepared,out)
    (out/'retry27/s.tasks.jsonl').write_text('')
    with pytest.raises(ValueError,match='Route changed'):split_source('s',prepared,out)
