#!/usr/bin/env bash

INPUT_DIR="$1"
OUTPUT_DIR="$2"

for file in "$INPUT_DIR"/*.json; do
    [ -f "$file" ] || continue

    echo "Processing file: $file"

    # strip file to basename without extension
    base_name=$(basename "$file" .json)

    output_file="$OUTPUT_DIR/$base_name.ll"
    output_mlir="$OUTPUT_DIR/$base_name.mlir"

    ./build/bin/bril2mlir < "$file" 2>&1 | tee "$output_mlir" | ./build/bin/bril-opt --pass-pipeline="builtin.module(convert-bril-to-std,rename-main-function,convert-arith-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,finalize-memref-to-llvm,canonicalize,cse)" - | ~/opt/llvm/bin/mlir-translate --mlir-to-llvmir - -o "$output_file"

done