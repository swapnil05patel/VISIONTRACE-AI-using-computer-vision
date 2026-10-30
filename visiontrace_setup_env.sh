#!/usr/bin/env bash

# ------------------------------------------------------------
# VisionTrace – Development environment bootstrap
# ------------------------------------------------------------
# 1. Node version (uses nvm if present)
if command -v nvm >/dev/null; then
  echo "Using nvm to select Node version..."
  nvm use || { echo "Failed to switch Node version via nvm"; exit 1; }
else
  echo "nvm not found – ensure Node version is installed manually"
fi

# 2. Install Node workspace dependencies (pnpm)
echo "Installing workspace Node dependencies..."
pnpm install || { echo "pnpm install failed"; exit 1; }

# 3. Python environment – prefer uv, fallback to venv
PYTHON_EXEC=$(grep '^PYTHON_EXECUTABLE=' .env.example | cut -d'=' -f2 | tr -d '\r')
PYTHON_EXEC=${PYTHON_EXEC:-python3}
if command -v uv >/dev/null; then
  echo "Creating Python env with uv..."
  uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt
else
  echo "Creating Python env with venv..."
  ${PYTHON_EXEC} -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
fi

# 4. Ensure .env exists – copy example if missing
if [ ! -f .env ]; then
  echo "Copying .env.example to .env"
  cp .env.example .env
fi

# 5. Create required storage directories
mkdir -p storage/dev && chmod 750 storage/dev

# 6. Verify model weight placeholder directory exists
mkdir -p ml/models

echo "Setup complete. You can now run ./start_all.sh to launch services."
