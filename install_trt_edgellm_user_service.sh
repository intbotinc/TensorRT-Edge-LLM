#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="trt-edgellm.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_SERVICE="${SCRIPT_DIR}/${SERVICE_NAME}"
USER_SYSTEMD_DIR="${HOME}/.config/systemd/user"
DST_SERVICE="${USER_SYSTEMD_DIR}/${SERVICE_NAME}"

if ! command -v systemctl >/dev/null 2>&1; then
  echo "Error: systemctl not found." >&2
  exit 1
fi

if [[ ! -f "${SRC_SERVICE}" ]]; then
  echo "Error: service file not found: ${SRC_SERVICE}" >&2
  exit 1
fi

mkdir -p "${USER_SYSTEMD_DIR}"
cp "${SRC_SERVICE}" "${DST_SERVICE}"

systemctl --user daemon-reload
systemctl --user enable --now "${SERVICE_NAME}"

echo "Installed and started ${SERVICE_NAME}"
systemctl --user status "${SERVICE_NAME}" --no-pager || true
echo "View logs: journalctl --user -u ${SERVICE_NAME} -f"
