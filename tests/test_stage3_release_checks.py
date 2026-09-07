from copy import deepcopy
import pytest
from data.stage3_release_checks import validate_base_recovery


def reports():
    return ([{'partition': 0, 'counts': dict(input_rows=10, packed_rows=5,
        rejected_processing=3, over_32768=2)}],
        [{'partition': 0, 'counts': dict(scanned=10, candidates=5,
            recovered_rows=1, packed_rows=1, over_32768=1, not_recoverable=3)}])


def test_recovery_counts_input_only_once_and_reclassifies_skips():
    summary = validate_base_recovery(*reports(), partitions=1)
    assert summary == dict(input_rows=10, packed_rows=6, recovered_rows=1,
        rejected_processing=1, over_32768=3, under_18=0)


@pytest.mark.parametrize('key,value', [('scanned', 9), ('packed_rows', 0),
    ('recovered_rows', 4), ('candidates', 6)])
def test_recovery_rejects_inconsistent_counts(key, value):
    base, recovery = deepcopy(reports())
    recovery[0]['counts'][key] = value
    with pytest.raises(ValueError):
        validate_base_recovery(base, recovery, partitions=1)


def test_recovery_requires_complete_unique_partitions():
    base, recovery = reports()
    with pytest.raises(ValueError):
        validate_base_recovery(base, recovery, partitions=2)
    with pytest.raises(ValueError):
        validate_base_recovery(base, recovery*2, partitions=1)
