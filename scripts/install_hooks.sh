#!/usr/bin/env bash
# Installs the pre-commit hook that refuses commits containing absolute local paths.
set -euo pipefail
printf "#!/usr/bin/env bash
exec python scripts/check_paths.py
" > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
echo "pre-commit hook installed"
