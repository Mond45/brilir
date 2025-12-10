//===- BrilDialect.cpp - Bril dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bril/BrilDialect.h"
#include "bril/BrilOps.h"

using namespace mlir;
using namespace mlir::bril;

#include "bril/BrilOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Bril dialect.
//===----------------------------------------------------------------------===//

void BrilDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bril/BrilOps.cpp.inc"
      >();
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

void ConstantOp::build(OpBuilder &builder, OperationState &state,
                         int64_t value) {
  state.addAttribute("value", builder.getI64IntegerAttr(value));
  state.addTypes(builder.getIntegerType(64));
}

void ConstantOp::build(OpBuilder &builder, OperationState &state,
                         bool value) {
  state.addAttribute("value", builder.getBoolAttr(value));
  state.addTypes(builder.getIntegerType(1));
}