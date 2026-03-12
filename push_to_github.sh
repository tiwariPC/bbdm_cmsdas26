#!/bin/bash
# Push this project to https://github.com/tiwariPC/bbdm_cmsdas26
set -e
cd "$(dirname "$0")"

if [ ! -d .git ]; then
  git init
fi

git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/tiwariPC/bbdm_cmsdas26.git

git add -A
git status

# Commit if there are staged changes (e.g. initial commit or new edits)
if ! git diff --staged --quiet 2>/dev/null; then
  git commit -m "Initial commit: CMS DAS 2026 BBDM materials"
else
  echo "No new changes to commit."
fi

git branch -M main
git push -u origin main

echo "Done. Repo: https://github.com/tiwariPC/bbdm_cmsdas26"
