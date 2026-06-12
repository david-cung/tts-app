#!/usr/bin/env bash
set -euo pipefail

APP_NAME="VieNeu-TTS"
BUNDLE_ID="com.vieneu.tts.launcher"
DEFAULT_PORT="${GRADIO_SERVER_PORT:-7860}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
DIST_DIR="${PROJECT_ROOT}/dist"
APP_DIR="${DIST_DIR}/${APP_NAME}.app"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
LAUNCHER="${MACOS_DIR}/${APP_NAME}"

if [[ ! -x "${PROJECT_ROOT}/.venv/bin/vieneu-web" ]]; then
  cat >&2 <<MSG
Missing ${PROJECT_ROOT}/.venv/bin/vieneu-web

Create the local environment first, then rerun this script:
  /opt/homebrew/bin/python3.12 -m venv .venv
  .venv/bin/python -m pip install -e .
MSG
  exit 1
fi

rm -rf "${APP_DIR}"
mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

cat > "${CONTENTS_DIR}/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>${APP_NAME}</string>
  <key>CFBundleExecutable</key>
  <string>${APP_NAME}</string>
  <key>CFBundleIdentifier</key>
  <string>${BUNDLE_ID}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>${APP_NAME}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>3.0.4</string>
  <key>CFBundleVersion</key>
  <string>3.0.4</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

cat > "${LAUNCHER}" <<LAUNCHER
#!/usr/bin/env bash
set -euo pipefail

APP_EXEC_DIR="\$(cd "\$(dirname "\$0")" && pwd -P)"
REL_PROJECT_ROOT="\$(cd "\${APP_EXEC_DIR}/../../../.." && pwd -P)"
BUILD_PROJECT_ROOT="${PROJECT_ROOT}"

if [[ -x "\${REL_PROJECT_ROOT}/.venv/bin/vieneu-web" ]]; then
  PROJECT_ROOT="\${REL_PROJECT_ROOT}"
else
  PROJECT_ROOT="\${BUILD_PROJECT_ROOT}"
fi

LOG_DIR="\${PROJECT_ROOT}/logs"
mkdir -p "\${LOG_DIR}"
LOG_FILE="\${LOG_DIR}/vieneu-tts-app.log"
BASE_PORT="${DEFAULT_PORT}"
HOST="127.0.0.1"

notify_error() {
  local message="\$1"
  /usr/bin/osascript -e "display dialog \\"\${message}\\" buttons {\\"OK\\"} default button \\"OK\\" with title \\"VieNeu-TTS\\"" >/dev/null 2>&1 || true
}

if [[ ! -x "\${PROJECT_ROOT}/.venv/bin/vieneu-web" ]]; then
  notify_error "Không tìm thấy môi trường chạy tại \${PROJECT_ROOT}/.venv. Hãy cài dependencies rồi build lại app."
  exit 1
fi

pick_port() {
  local port
  for port in "\${BASE_PORT}" 7861 7862 7863 7864 7865 7866 7867 7868 7869 7870; do
    if /usr/bin/curl -fsS "http://\${HOST}:\${port}" >/dev/null 2>&1; then
      echo "\${port}"
      return 0
    fi
    if /usr/bin/python3 - "\${HOST}" "\${port}" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
try:
    sock.bind((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
    then
      echo "\${port}"
      return 0
    fi
  done
  return 1
}

PORT="\$(pick_port)" || {
  notify_error "Không tìm được port trống từ 7860 đến 7870."
  exit 1
}

URL="http://\${HOST}:\${PORT}"

if /usr/bin/curl -fsS "\${URL}" >/dev/null 2>&1; then
  [[ "\${VIENEU_TTS_OPEN_BROWSER:-1}" == "1" ]] && /usr/bin/open "\${URL}"
  exit 0
fi

{
  echo "==== \$(date) ===="
  echo "Starting VieNeu-TTS from \${PROJECT_ROOT}"
  echo "URL: \${URL}"
} >> "\${LOG_FILE}"

cd "\${PROJECT_ROOT}"
GRADIO_SERVER_NAME="\${HOST}" GRADIO_SERVER_PORT="\${PORT}" GRADIO_SHARE="0" \
  "\${PROJECT_ROOT}/.venv/bin/vieneu-web" >> "\${LOG_FILE}" 2>&1 &

SERVER_PID="\$!"

for _ in {1..90}; do
  if /usr/bin/curl -fsS "\${URL}" >/dev/null 2>&1; then
    [[ "\${VIENEU_TTS_OPEN_BROWSER:-1}" == "1" ]] && /usr/bin/open "\${URL}"
    exit 0
  fi
  if ! /bin/kill -0 "\${SERVER_PID}" >/dev/null 2>&1; then
    notify_error "VieNeu-TTS khởi động thất bại. Xem log: \${LOG_FILE}"
    exit 1
  fi
  /bin/sleep 1
done

notify_error "VieNeu-TTS khởi động quá lâu. Xem log: \${LOG_FILE}"
exit 1
LAUNCHER

chmod +x "${LAUNCHER}"

echo "Built ${APP_DIR}"
echo "Open it with:"
echo "  open '${APP_DIR}'"
