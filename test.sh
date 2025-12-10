#!/usr/bin/env bash

INPUT_DIR="inputs"
PROGRAM="build/bin/bril2mlir"
OUTPUT_FILE="results.csv"

: > "$OUTPUT_FILE"   # truncate output file

for file in "$INPUT_DIR"/*.json; do
    [ -f "$file" ] || continue

    echo "Testing file: $file"

    "$PROGRAM" < "$file" > /dev/null 2>&1
    rc=$?

    printf "%s,%d\n" "$(basename "$file")" "$rc" >> "$OUTPUT_FILE"
done