#!/bin/bash
# One-time setup on macOS:   bash setup_mac.sh
set -e
cd "$(dirname "$0")"
if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 not found. Install it with:  brew install python@3.11"
  exit 1
fi
if [ ! -d venv ]; then
  python3.11 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import mediapipe, cv2, torch, onnxruntime; print('all libraries OK')"
echo ""
echo "Setup finished. Every new Terminal:"
echo "  cd $(pwd)"
echo "  source venv/bin/activate"
echo "Then check everything with:  python selftest.py"
