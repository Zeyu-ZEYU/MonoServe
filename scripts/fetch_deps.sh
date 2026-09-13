#!/bin/bash
# Fetch the pinned third-party sources the CUDA code builds against.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/third_party"
cd "$ROOT/third_party"

fetch() {   # name url commit
  if [ ! -d "$1/.git" ]; then
    git clone --filter=blob:none "$2" "$1"
  fi
  git -C "$1" fetch --depth 1 origin "$3" 2>/dev/null || git -C "$1" fetch origin
  git -C "$1" checkout --quiet "$3"
  echo "$1 at $(git -C "$1" rev-parse --short HEAD)"
}

fetch cutlass https://github.com/NVIDIA/cutlass.git 147295a
fetch ThunderKittens https://github.com/HazyResearch/ThunderKittens.git 4d96999
