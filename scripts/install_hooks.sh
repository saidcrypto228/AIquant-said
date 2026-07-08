#!/usr/bin/env sh
# One-time setup: point git at the tracked hooks directory so the pre-commit
# hook keeps AIQuant_Colab.ipynb in sync with scripts/build_colab.py automatically.
# Run once per clone:  sh scripts/install_hooks.sh
set -e
git config core.hooksPath scripts/hooks
chmod +x scripts/hooks/pre-commit
echo "✓ Git hooks installed (core.hooksPath = scripts/hooks)"
