#!/bin/bash

SEARCH_DIR=${1}

log_time() {
    date "+%Y-%m-%d %H:%M:%S"
}

echo "$(log_time) [INFO] Refreshing access time for files in: ${SEARCH_DIR}"
echo "--------------------------------------------------"

find "${SEARCH_DIR}" -type f -print -exec sh -c 'head -n 1 "$1" > /dev/null' _ {} \;

echo "--------------------------------------------------"
echo "$(log_time) [INFO] Finished refreshing access time."