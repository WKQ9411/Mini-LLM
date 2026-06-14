#!/usr/bin/env bash
set -euo pipefail

frontend_port="${1:-}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
backend_port=12345

is_port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :$port" | grep -q LISTEN
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1
  fi
}

while is_port_in_use "$backend_port"; do
  backend_port=$((backend_port + 1))
done

venv_python="$repo_root/.venv/bin/python"
if [[ ! -x "$venv_python" ]]; then
  echo "Python venv not found or not executable: $venv_python" >&2
  exit 1
fi

echo "[start] backend port: $backend_port"
echo "[start] starting backend..."
"$venv_python" -m uvicorn architecture_lab.backend.main:app --reload --host 127.0.0.1 --port "$backend_port" --log-level info &
backend_pid=$!

# 等待后端端口就绪（最多 30 秒）
echo "[start] waiting for backend to be ready..."
waited=0
max_wait=30
backend_ready=false
while [ "$waited" -lt "$max_wait" ]; do
  if ! kill -0 "$backend_pid" >/dev/null 2>&1; then
    wait "$backend_pid" || true
    echo "backend exited early" >&2
    exit 1
  fi
  if is_port_in_use "$backend_port"; then
    backend_ready=true
    break
  fi
  sleep 1
  waited=$((waited + 1))
done
if [ "$backend_ready" = false ]; then
  echo "backend did not become ready within $max_wait seconds" >&2
  exit 1
fi
echo "[start] backend is ready"

stop_backend() {
  if kill -0 "$backend_pid" >/dev/null 2>&1; then
    echo "[start] stopping backend..."
    kill "$backend_pid" >/dev/null 2>&1 || true
    wait "$backend_pid" 2>/dev/null || true
  fi
}

trap stop_backend EXIT INT TERM

export VITE_BACKEND_HOST=127.0.0.1
export VITE_BACKEND_PORT="$backend_port"

echo "[start] starting frontend dev server..."
vite_args=(vite --host 127.0.0.1)
if [[ -n "$frontend_port" ]]; then
  echo "[start] frontend port: $frontend_port"
  echo "[start] open: http://localhost:$frontend_port"
  vite_args+=(--port "$frontend_port")
else
  echo "[start] frontend port: Vite default (usually 5173)"
  echo "[start] open: check the Vite URL printed below"
fi

cd "$script_dir/frontend"
npx "${vite_args[@]}"
