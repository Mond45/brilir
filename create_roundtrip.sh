#!/usr/bin/env bash

INPUT_DIR="inputs"
OUTPUT_DIR="outputs"

for file in "$INPUT_DIR"/*.json; do
    [ -f "$file" ] || continue

    echo "Processing file: $file"


    # strip file to basename without extension
    base_name=$(basename "$file" .json)

    output_file="$OUTPUT_DIR/$base_name.bril"

    ./build/bin/bril2mlir < "$file" 2>&1 | ./build/bin/mlir2bril | bril2txt > "$output_file"

done

# Run fix.sh in outputs dir to add ARGS