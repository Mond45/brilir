# An out-of-tree MLIR dialect for Bril

This repo contains an out-of-tree [MLIR](https://mlir.llvm.org/) dialect for [Bril](https://capra.cs.cornell.edu/bril/intro.html) a standalone `opt`-like tool to operate on Bril dialect, and conversion tools for translating between Bril and MLIR (`bril2mlir` and `mlir2bril`).

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`:

```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DLLVM_CCACHE_BUILD=ON \
   -DCMAKE_INSTALL_PREFIX=$HOME/opt/llvm \
   -DLLVM_INSTALL_UTILS=ON
cmake --build . --target install
```

To build `brilir`:

```sh
export BUILD_DIR=$HOME/repos/llvm-project/build
export PREFIX=$HOME/opt/llvm
mkdir build
cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLVM_CCACHE_BUILD=ON
cmake --build .
```

## Example Usage

```sh
bril2mlir < bril_input.json 2>&1 | mlir2bril | bril2txt

bril2mlir < bril_input.json 2>&1 | bril-opt --pass-pipeline="builtin.module(convert-bril-to-std,rename-main-function,convert-arith-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,finalize-memref-to-llvm,canonicalize,cse)" - | mlir-translate --mlir-to-llvmir - -o output.ll
clang++ output.ll main.cpp
```
