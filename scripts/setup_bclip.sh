#!/usr/bin/env bash
set -euo pipefail
mkdir -p external
if [ ! -d external/B-CLIP/.git ]; then
  git clone --branch v1 https://github.com/fzohra/B-CLIP.git external/B-CLIP
else
  echo "external/B-CLIP already exists"
fi
cat <<'MSG'
β-CLIP cloned to external/B-CLIP.
Now follow external/B-CLIP/INSTALL.md and download the checkpoint you want to use.
MSG
