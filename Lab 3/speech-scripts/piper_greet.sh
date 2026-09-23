#!/usr/bin/env bash
# Greet Abiola Bolaji using the sw_CD-lanfrica-medium Piper voice.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$SCRIPT_DIR/sw_CD-lanfrica-medium.onnx"

python3 -m piper \
  --model "$MODEL" \
  --output-file greeting.wav \
  -- "Hello, Oga,Prime Minister, Excellency, Importer, Exporter, Pure Water, Sir Abiola Bolaji. Welcome!"
aplay greeting.wav
