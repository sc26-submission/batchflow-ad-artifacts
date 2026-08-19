#!/usr/bin/env bash
set -euo pipefail

WORKLOAD_NAME="imagenet_1j_resnet18"
DEPLOYMENT_NAME="colocated"
POLICY_NAME="full"
EXPERIMENT_ARGS=()

for arg in "$@"; do
  case "${arg}" in
    workload=*)
      WORKLOAD_NAME="${arg#workload=}"
      EXPERIMENT_ARGS+=("${arg}")
      ;;
    deployment=*)
      DEPLOYMENT_NAME="${arg#deployment=}"
      ;;
    policy=*)
      POLICY_NAME="${arg#policy=}"
      ;;
    *)
      EXPERIMENT_ARGS+=("${arg}")
      ;;
  esac
done

DATASET_NAME="$(python - "${WORKLOAD_NAME}" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

workload_name = sys.argv[1]
path = Path("experiments/config/workload") / f"{workload_name}.yaml"
if not path.is_file():
    raise SystemExit(f"Unknown workload config: {path}")

cfg = OmegaConf.load(path)
print(cfg.dataset)
PY
)"

BATCHFLOW_ADDR="127.0.0.1:50051"
LOG_DIR="logs/experiment_runtime"
mkdir -p "${LOG_DIR}"

echo "Starting BatchFlow workload=${WORKLOAD_NAME} dataset=${DATASET_NAME} deployment=${DEPLOYMENT_NAME} policy=${POLICY_NAME}"
python -m batchflow.deployment.launch_batchflow \
  dataset="${DATASET_NAME}" \
  deployment="${DEPLOYMENT_NAME}" \
  policy="${POLICY_NAME}" \
  > "${LOG_DIR}/batchflow.out" \
  2> "${LOG_DIR}/batchflow.err" &

BATCHFLOW_PID=$!

cleanup() {
  echo "Stopping BatchFlow..."
  kill "${BATCHFLOW_PID}" 2>/dev/null || true
  wait "${BATCHFLOW_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python - <<'PY'
import socket
import sys
import time

host = "127.0.0.1"
port = 50051
deadline = time.time() + 60

while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1):
            print("Coordinator is ready")
            sys.exit(0)
    except OSError:
        time.sleep(0.5)

print(f"Timed out waiting for {host}:{port}", file=sys.stderr)
sys.exit(1)
PY

echo "Running experiment..."
python -m experiments.run_experiment \
  system=batchflow \
  system.client.coordinator_address="${BATCHFLOW_ADDR}" \
  "${EXPERIMENT_ARGS[@]}"
