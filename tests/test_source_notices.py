from data.source_notices import notice_files, copy_notice, validate_notice_bundle
import pytest


def test_notice_selection_ignores_cache_and_data(tmp_path):
    for name in ('README.md','LICENSE','rows.jsonl','.cache/LICENSE','nested/NOTICE','nested/README.md','LICENSES/Apache-2.0.txt'):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('source')
    assert {str(p.relative_to(tmp_path)) for p in notice_files(tmp_path)}=={
        'README.md','LICENSE','nested/NOTICE','LICENSES/Apache-2.0.txt'}


def test_notice_collection_does_not_follow_file_symlink(tmp_path):
    root=tmp_path/'source';root.mkdir()
    outside=tmp_path/'outside';outside.write_text('not source content')
    (root/'LICENSE').symlink_to(outside)
    assert list(notice_files(root))==[]


def test_copied_notices_are_exact_and_existing_conflicts_fail(tmp_path):
    source=tmp_path/'source';source.mkdir();path=source/'LICENSE';path.write_bytes(b'Original\r\n')
    target=tmp_path/'release'
    record=copy_notice(path,source,target)
    assert (target/'LICENSE').read_bytes()==b'Original\r\n' and record['bytes']==10
    path.write_bytes(b'Changed')
    with pytest.raises(ValueError):copy_notice(path,source,target)


def bundle(tmp_path):
    source=tmp_path/'upstream';source.mkdir();path=source/'LICENSE';path.write_text('Original')
    root=tmp_path/'bundle'
    file=copy_notice(path,source,root/'agents'/'source')
    index={'status':'collected','sources':[{'component':'agents','key':'source',
        'source_id':'org/source','revision':'revision','files':[file]}]}
    return root,index,{('agents','source'):('org/source','revision')}


def test_bundle_validates_exact_source_bytes(tmp_path):
    validate_notice_bundle(*bundle(tmp_path))


@pytest.mark.parametrize('failure',['bytes','revision','coverage','path'])
def test_bundle_rejects_tampering_or_missing_source(tmp_path,failure):
    root,index,expected=bundle(tmp_path)
    if failure=='bytes':(root/'agents/source/LICENSE').write_text('Changed')
    elif failure=='revision':index['sources'][0]['revision']='main'
    elif failure=='coverage':index['sources']=[]
    else:index['sources'][0]['files'][0]['upstream_path']='../LICENSE'
    with pytest.raises(ValueError):validate_notice_bundle(root,index,expected)
