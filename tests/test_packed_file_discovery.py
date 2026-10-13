import pytest
from data.packed_file_discovery import discover_packed_parquet_files


def touch(root,name):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch();return str(path)


def test_release_discovers_all_components_but_not_raw_or_partial_files(tmp_path):
    expected=[touch(tmp_path,f'data/{kind}/part-000/shard.parquet')
        for kind in ('base','base_recovery','agents','expansion')]
    for extra in ('raw/base/shard.parquet','quarantine/base/shard.parquet',
                  'data/base/in-progress/shard.parquet'):
        touch(tmp_path,extra)
    assert discover_packed_parquet_files(tmp_path)==sorted(expected)


def test_flat_layout_still_works(tmp_path):
    expected=touch(tmp_path,'shard.parquet')
    assert discover_packed_parquet_files(tmp_path)==[expected]


def test_conflicting_layouts_raise_instead_of_dropping_one_component(tmp_path):
    touch(tmp_path,'shard.parquet');touch(tmp_path,'data/agents/part-000/shard.parquet')
    with pytest.raises(ValueError,match='Ambiguous'):discover_packed_parquet_files(tmp_path)


def test_empty_folder_raises(tmp_path):
    with pytest.raises(ValueError,match='No packed'):discover_packed_parquet_files(tmp_path)
