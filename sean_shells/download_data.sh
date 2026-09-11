#!/usr/bin/env bash

hf download leonli66/stage3-final-mixture-20260910-packed --repo-type dataset --revision f46f4124d6c602ec74d931d6999def02b6d9e7bb --local-dir /p/vast1/rl/compression_data/packed_batches/stage3-final-mixture-20260910-packed --include "packed-cs16-16384/**" "packed-cs16-32768/**" release.json UPLOAD_STATUS.json
