#!/usr/bin/env bash
# Compress all .json files in the current directory as .xz

for f in *.json; do
    if [ -f "$f" ]; then
        xz -k -z "$f"
    fi
done
