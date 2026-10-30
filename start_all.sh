#!/usr/bin/env bash

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Load .env
if [ -f "$DIR/.env" ]; then
  export $(grep -v '^#' "$DIR/.env" | xargs)
else
  echo ".env not found – copying from .env.example"
  cp "$DIR/.env.example" "$DIR/.env"
  export $(grep -v '^#' "$DIR/.env" | xargs)
fi

export PORT=${PORT:-8080}
export VITE_PORT=3000
export BASE_PATH=${BASE_PATH:-/}

port_in_use() { lsof -i TCP:$1 -s TCP:LISTEN -t >/dev/null 2>&1; }

# Try starting Postgres container if docker is accessible
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  if ! docker ps --filter "name=visiontrace_pg" --filter "status=running" -q | grep -q .; then
    echo "Starting PostgreSQL container via Docker..."
    docker start visiontrace_pg 2>/dev/null || docker run -d --name visiontrace_pg -e POSTGRES_USER=visiontrace -e POSTGRES_PASSWORD=visiontrace -e POSTGRES_DB=visiontrace -p 5432:5432 postgres:16-alpine || true
  fi
fi

# Express backend (port 8080)
if ! port_in_use "$PORT"; then
  echo "Starting Express API backend on port $PORT..."
  pnpm --filter @workspace/api-server run dev &
else
  echo "Backend port $PORT is already in use"
fi

# React frontend (port 3000)
if ! port_in_use "$VITE_PORT"; then
  echo "Starting React frontend on port $VITE_PORT..."
  PORT="$VITE_PORT" BASE_PATH="$BASE_PATH" pnpm --filter @workspace/visiontrace run dev &
else
  echo "Frontend port $VITE_PORT is already in use"
fi

wait
