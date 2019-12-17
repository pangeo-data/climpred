#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (black, flake8, isort)"

source activate climpred-dev

echo "[flake8]"
flake8 climpred --max-line-length=88 --exclude=__init__.py --ignore=W605,W503,C901

echo "[black]"
black --check --line-length=88 -S climpred

echo "[isort]"
isort --recursive -w 88 --check-only climpred
