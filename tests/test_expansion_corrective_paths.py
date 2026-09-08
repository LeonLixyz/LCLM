import pytest
from data.audit_full_expansion_modal import GENERATED, AUDITS, audit_paths
from data.full_expansion_rollouts import generate_all


def empty_run(root, provenance=None):
    root.mkdir(exist_ok=True)
    (root / 'synthetic.build.json').write_text('{}')
    (root / 'synthetic.tasks.jsonl').write_text('')
    return generate_all(None, 'test-model', 'test-revision', root, lambda: None,
                        sources=['synthetic'], input_provenance=provenance)


def test_legacy_manifest_unchanged_with_optional_provenance(tmp_path):
    original = empty_run(tmp_path / 'legacy')['manifest']
    provenance = {'source': 'multidoc2dial', 'tasks_sha256': 'test-hash'}
    corrective = empty_run(tmp_path / 'repair', provenance)['manifest']
    assert corrective.pop('input_provenance') == provenance
    assert corrective == original
    assert 'input_provenance' not in original


def test_corrective_resume_rejects_changed_input_binding(tmp_path):
    empty_run(tmp_path, {'tasks_sha256': 'first'})
    with pytest.raises(RuntimeError, match='Incompatible resume'):
        empty_run(tmp_path, {'tasks_sha256': 'changed'})


def test_audit_default_and_corrective_paths():
    assert audit_paths() == (GENERATED, AUDITS)
    root = GENERATED.parent / 'multidoc2dial-chronological-v1-pilot'
    assert audit_paths(str(root)) == (root, root / 'format-audit')


@pytest.mark.parametrize('path', [str(GENERATED), '/tmp/elsewhere', '/', str(GENERATED / 'nested')])
def test_audit_rejects_wrong_destinations(path):
    with pytest.raises(ValueError):
        audit_paths(path)
