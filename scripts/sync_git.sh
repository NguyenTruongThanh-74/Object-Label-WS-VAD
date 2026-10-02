#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: bash scripts/sync_git.sh push [commit message]"
  echo "       bash scripts/sync_git.sh pull"
}

action="${1:-}"
if [[ "$action" != "push" && "$action" != "pull" ]]; then
  usage
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/.." rev-parse --show-toplevel)"
cd "$repo_root"

branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
  echo "Error: detached HEAD; check out a branch before syncing." >&2
  exit 1
fi

echo "Repository: $repo_root"
echo "Branch: $branch"
git status --short --branch

if [[ "$action" == "pull" ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working tree is not clean. Commit or stash changes before pulling." >&2
    exit 1
  fi
  if ! git pull --rebase; then
    echo "Pull stopped. Resolve conflicts, then continue the rebase or run: git rebase --abort" >&2
    exit 1
  fi
  git status --short --branch
  exit 0
fi

read -r -p "Stage all non-ignored changes in this repository? [y/N] " answer
if [[ ! "$answer" =~ ^[Yy]$ ]]; then
  echo "Canceled; nothing was staged."
  exit 0
fi

git add -A
if ! git diff --cached --quiet; then
  git diff --cached --stat
  message="${2:-Update project}"
  read -r -p "Commit these staged changes and push to origin/$branch? [y/N] " answer
  if [[ ! "$answer" =~ ^[Yy]$ ]]; then
    echo "Canceled; changes remain staged and were not pushed."
    exit 0
  fi
  git commit -m "$message"
fi

git push --set-upstream origin "$branch"
git status --short --branch
