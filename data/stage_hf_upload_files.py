import concurrent.futures, hashlib, json, os, time
from pathlib import Path
started = time.time()
for local in Path('/tmp').glob('hf-*-*'):
    if not local.is_dir() or local.name.split('-')[1] not in ('raw','packed'):
        continue
    kind = local.name.split('-')[1]
    release = json.loads(Path('/data/stage3-build-20260906/hf-publication-20260911-v1',kind,'files','release.json').read_text())
    entries = release['files']
    pending = []
    skipped = 0
    for path in sorted(local.rglob('*.parquet')):
        if not path.is_symlink():
            continue
        name = path.relative_to(local).as_posix()
        metadata = local/'.cache/huggingface/upload'/f'{name}.metadata'
        lines = metadata.read_text().splitlines() if metadata.exists() else []
        if len(lines) >= 8 and lines[6] == '1':
            skipped += 1
            continue
        pending.append((path, entries[name]))
    print(json.dumps({'event':'scratch_copy_start','queue':local.name,'pending_files':len(pending),'already_uploaded_skipped':skipped,'bytes':sum(r['bytes'] for p,r in pending)}),flush=True)
    def copy_one(item):
        path, expected = item
        source = path.resolve()
        stats = source.stat()
        temporary = path.with_name(path.name+'.copying')
        digest = hashlib.sha256()
        size = 0
        with source.open('rb') as reader, temporary.open('wb') as writer:
            while chunk := reader.read(8*1024*1024):
                digest.update(chunk); writer.write(chunk); size += len(chunk)
        if size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
            temporary.unlink()
            raise ValueError('Source hash mismatch: '+str(path))
        os.utime(temporary,ns=(stats.st_atime_ns,stats.st_mtime_ns))
        os.replace(temporary,path)
        return size
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        copied = sum(pool.map(copy_one,pending))
    print(json.dumps({'event':'scratch_copy_complete','queue':local.name,'files':len(pending),'bytes':copied,'seconds':round(time.time()-started,1)}),flush=True)
