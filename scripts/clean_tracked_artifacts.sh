#!/usr/bin/env bash
set -euo pipefail

# Use before opening a PR if local artifacts were committed in the past.
# It removes artifacts from the Git index only; files remain on your disk.

git rm --cached .env 2>/dev/null || true
git rm -r --cached data results results_user results_simulator models models_user models_simulator 2>/dev/null || true
git add .gitignore .env.example

echo "Done. Review with: git status"
echo "Then commit with: git commit -m 'chore: exclude local artifacts from git'"
