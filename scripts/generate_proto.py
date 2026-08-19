#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    # Assumes this file lives in <repo>/scripts/generate_proto.py
    root_dir = Path(__file__).resolve().parent.parent
    proto_dir = root_dir / "batchflow" / "proto"
    out_dir = root_dir / "batchflow" / "proto"
    proto_file = proto_dir / "batchflow.proto"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not proto_file.exists():
        print(f"Proto file not found: {proto_file}", file=sys.stderr)
        return 1

    print(f"Using Python: {sys.executable}")

    try:
        import grpc_tools  # noqa: F401
    except ImportError:
        print(
            "grpcio-tools is not installed in this Python environment.\n"
            "Install it with:\n"
            f'  "{sys.executable}" -m pip install grpcio-tools',
            file=sys.stderr,
        )
        return 1

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto_file),
    ]

    print("Running:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"protoc generation failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    init_file = out_dir / "__init__.py"
    init_file.touch(exist_ok=True)

    print(f"Generated into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())