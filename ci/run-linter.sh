#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (black, flake8, isort)"

echo "[flake8]"
flake8 climpred --exclude=__init__.py

echo "[black]"
black --check climpred

echo "[isort]"
isort --recursive --check-only climpred
