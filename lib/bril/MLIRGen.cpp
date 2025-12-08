//===- MLIRGen.cpp - MLIR Generation from a Bril JSON
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Bril JSON
// for the Bril language.
//
//===----------------------------------------------------------------------===//

#include "bril/MLIRGen.h"
#include "bril/BrilDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using namespace mlir::bril;
using namespace bril;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the Bril JSON.
///
/// This will emit operations that are specific to the Bril language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  mlir::OwningOpRef<mlir::ModuleOp> mlirGen(nlohmann::json &json) {
    // Create the module.
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

    // Set the builder insertion point to the module.
    builder.setInsertionPointToStart(module.getBody());

    return module;
  }

private:
  mlir::OpBuilder builder;
};

} // namespace

namespace bril {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          nlohmann::json &json) {
  return MLIRGenImpl(context).mlirGen(json);
}

} // namespace bril
