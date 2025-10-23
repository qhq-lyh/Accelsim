#!/bin/bash
# ====================================================
# Script: build_hotspot.sh
# Description: Clone and build HotSpot project
# Author: qhq-lyh
# ====================================================

set -e 

REPO_URL="https://github.com/qhq-lyh/HotSpot.git"

REPO_NAME=$(basename -s .git "$REPO_URL")

if [ -d "$REPO_NAME" ]; then
    echo "Directory '$REPO_NAME' already exists."
    echo "If you want to re-clone, delete it first:"
    echo "  rm -rf $REPO_NAME"
    exit 1
fi

echo "Cloning repository from $REPO_URL ..."
git clone "$REPO_URL"

cd "$REPO_NAME"

echo "Building project with make ..."
make -j$(nproc)

echo "Build completed successfully!"
