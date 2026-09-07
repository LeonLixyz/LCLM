"""Conservatively select upstream notice files without following links."""
import hashlib
from pathlib import Path


def notice_files(root):
    root = Path(root)
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in {'.git','.cache','__pycache__'} for part in relative.parts):
            continue
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            continue
        name = path.name.lower()
        selected = (name.startswith(('license','licence','notice','copying','copyright','authors'))
                    or any(part.lower() == 'licenses' for part in relative.parts[:-1])
                    or (path.parent == root and name.startswith('readme')))
        if selected:
            if path.stat().st_size > 5 * 1024 * 1024:
                raise ValueError(f'Notice needs manual size/content review: {relative}')
            yield path


def copy_notice(path, root, destination):
    relative = path.relative_to(root)
    payload = path.read_bytes()
    target = destination / relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise ValueError(f'Existing notice differs from pinned source: {target}')
    target.write_bytes(payload)
    return {'upstream_path':str(relative),'bytes':len(payload),
            'sha256':hashlib.sha256(payload).hexdigest()}


def validate_notice_bundle(root, index, expected_sources):
    """Verify coverage and bytes, not legal clearance of the underlying data."""
    actual = {(r['component'],r['key']):r for r in index['sources']}
    if (index.get('status') != 'collected' or len(actual) != len(index['sources'])
            or set(actual) != set(expected_sources)):
        raise ValueError('Source notice coverage mismatch')
    for key, (source_id, revision) in expected_sources.items():
        record = actual[key]
        if (record['source_id'],record['revision']) != (source_id,revision) or not record['files']:
            raise ValueError(f'Missing or stale source notices: {key}')
        seen = set()
        for file in record['files']:
            relative = Path(file['upstream_path'])
            if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
                raise ValueError('Unsafe or duplicate notice path')
            seen.add(str(relative))
            path = root / key[0] / key[1] / relative
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('Notice path escaped bundle')
            data = path.read_bytes()
            if len(data) != file['bytes'] or hashlib.sha256(data).hexdigest() != file['sha256']:
                raise ValueError('Notice bytes differ from collected upstream source')
