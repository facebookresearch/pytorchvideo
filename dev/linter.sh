#!/bin/bash -ev
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Run this script at project root with "./dev/linter.sh" before you commit.

echo "Running autoflake..."
python -m autoflake --remove-all-unused-imports -i .

echo "Running isort..."
isort -y -sp .

echo "Running black..."
black .

echo "Running flake8..."
if [ -x "$(command -v flake8)" ]; then
  flake8 .
else
  python3 -m flake8 .
fi

command -v arc > /dev/null && {
  echo "Running arc lint ..."
  arc lint
}
