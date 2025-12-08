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
#include "bril/BrilOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <nlohmann/json_fwd.hpp>
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

namespace llvm {
template <> struct DenseMapInfo<std::string> {
  static inline std::string getEmptyKey() { return ""; }

  static inline std::string getTombstoneKey() { return "\0"; }

  static unsigned getHashValue(const std::string &value) {
    return std::hash<std::string>()(value);
  }

  static bool isEqual(const std::string &lhs, const std::string &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

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
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (auto &funcJson : json["functions"]) {
      if (llvm::failed(mlirGenFunction(funcJson))) {
        theModule->emitError("failed to generate function");
        return nullptr;
      }
    }

    // TODO: uncomment this
    // if (llvm::failed(mlir::verify(theModule))) {
    //   theModule->emitError("module verification failed");
    //   return nullptr;
    // }

    return theModule;
  }

private:
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
  llvm::ScopedHashTable<std::string, mlir::Value> symbolTable;
  llvm::ScopedHashTable<std::string, mlir::Block *> labelToBlock;

  llvm::LogicalResult declare(std::string var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  mlir::Type getType(StringRef type) {
    if (type == "int")
      return builder.getIntegerType(32);
    if (type == "bool")
      return builder.getIntegerType(1);
    return nullptr;
  }

  std::vector<std::vector<nlohmann::json>>
  splitBlocks(nlohmann::json &instrsJson) {
    std::vector<std::vector<nlohmann::json>> blocks = {};
    std::vector<nlohmann::json> currentBlock = {};

    for (auto &instrJson : instrsJson) {
      if (instrJson.contains("op")) {
        currentBlock.push_back(instrJson);

        auto op = instrJson["op"].get<std::string>();

        if (op == "br" || op == "jmp" || op == "ret") {
          blocks.push_back(currentBlock);
          currentBlock = {};
        }
      } else {
        if (!currentBlock.empty()) {
          blocks.push_back(currentBlock);
        }

        currentBlock = {instrJson};
      }
    }

    return blocks;
  }

  llvm::LogicalResult mlirGenFunction(nlohmann::json &funcJson) {
    ScopedHashTableScope<std::string, mlir::Value> varScope(symbolTable);
    ScopedHashTableScope<std::string, mlir::Block *> labelScope(labelToBlock);

    auto funcName = funcJson["name"].get<std::string>();

    mlir::SmallVector<mlir::Type, 4> argTypes = {};
    for (auto &arg : funcJson["args"]) {
      auto argType = getType(arg["type"].get<std::string>());
      argTypes.push_back(argType);
    }

    mlir::TypeRange returnTypes = {};
    if (funcJson.contains("type")) {
      returnTypes = {getType(funcJson["type"].get<std::string>())};
    }

    builder.setInsertionPointToEnd(theModule.getBody());

    auto func = FuncOp::create(builder, builder.getUnknownLoc(), funcName,
                               builder.getFunctionType(argTypes, returnTypes));

    auto &entryBlock = func.front();
    for (auto nameValue :
         llvm::zip(funcJson["args"], entryBlock.getArguments())) {
      auto name = std::get<0>(nameValue)["name"].get<std::string>();
      auto value = std::get<1>(nameValue);

      if (llvm::failed(declare(name, value))) {
        func.emitError("failed to declare argument ") << name;
        return llvm::failure();
      }
    }

    builder.setInsertionPointToStart(&entryBlock);

    auto blocks = splitBlocks(funcJson["instrs"]);

    // TODO: first pass to create all blocks and map labels
    for (auto &block : blocks) {
      if (block.front().contains("label")) {

      } else {
      }
    }

    for (auto &block : blocks) {
      // if (block.front().contains("label")) {
      //   auto labelName = block.front()["label"].get<std::string>();
      //   auto *mlirBlock = func.addBlock();

      //   for (auto &instr : block) {
      //     if (instr.contains("op") && instr["op"] == "get") {
      //       auto blockArg = mlirBlock->addArgument(
      //           getType(instr["type"].get<std::string>()),
      //           builder.getUnknownLoc());
      //       if (llvm::failed(
      //               declare(instr["dest"].get<std::string>(), blockArg))) {
      //         func.emitError("failed to declare block argument ");
      //         return llvm::failure();
      //       }
      //     }
      //   }

      //   labelToBlock.insert(labelName, mlirBlock);
      //   builder.setInsertionPointToStart(mlirBlock);

      //   block.erase(block.begin());
      // } else {
      //   auto *mlirBlock = func.addBlock();
      //   builder.setInsertionPointToStart(mlirBlock);
      // }

      for (auto &instr : block) {
        if (llvm::failed(mlirGenInstruction(instr))) {
          func.emitError("failed to generate instruction");
          return llvm::failure();
        }
      }
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenInstruction(nlohmann::json &instrJson) {
    auto op = instrJson["op"].get<std::string>();

    if (op == "const") {
      return mlirGenConst(instrJson);
    } else if (op == "add" || op == "sub" || op == "mul" || op == "div" ||
               op == "eq" || op == "lt" || op == "gt" || op == "le" ||
               op == "ge" || op == "and" || op == "or") {
      return mlirGenBinaryOp(instrJson);

    } else if (op == "undef") {
      return mlirGenUndef(instrJson);
    } else if (op == "id") {
      return mlirGenId(instrJson);
    } else if (op == "not") {
      return mlirGenNot(instrJson);
    } else if (op == "br") {
      return mlirGenBranch(instrJson);
    } else if (op == "jmp") {
      return mlirGenJmp(instrJson);
    } else if (op == "ret") {
      return mlirGenRet(instrJson);
    } else if (op == "set" || op == "get") {
    } else if (op == "print") {
      return mlirGenPrint(instrJson);
    } else if (op == "nop") {
      return mlirGenNop(instrJson);
    } else {
      llvm::errs() << "Unhandled operation: " << op << "\n";
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenConst(nlohmann::json &instrJson) {
    auto dest = instrJson["dest"].get<std::string>();
    if (instrJson["type"] == "int") {
      int32_t value = instrJson["value"].get<int32_t>();
      auto constOp =
          ConstantOp::create(builder, builder.getUnknownLoc(), value);
      if (llvm::failed(declare(dest, constOp.getResult()))) {
        return mlir::failure();
      }
    } else if (instrJson["type"] == "bool") {
      bool value = instrJson["value"].get<bool>();
      auto constOp =
          ConstantOp::create(builder, builder.getUnknownLoc(), value);
      if (llvm::failed(declare(dest, constOp.getResult()))) {
        return mlir::failure();
      }
    } else {
      return mlir::failure();
    }
    return llvm::success();
  }

  llvm::LogicalResult mlirGenUndef(nlohmann::json &instrJson) {
    auto dest = instrJson["dest"].get<std::string>();
    if (instrJson["type"] == "int") {
      auto undefOp = ConstantOp::create(builder, builder.getUnknownLoc(), 0);
      if (llvm::failed(declare(dest, undefOp.getResult()))) {
        return mlir::failure();
      }
    } else if (instrJson["type"] == "bool") {
      auto undefOp =
          ConstantOp::create(builder, builder.getUnknownLoc(), false);
      if (llvm::failed(declare(dest, undefOp.getResult()))) {
        return mlir::failure();
      }
    } else {
      return mlir::failure();
    }
    return llvm::success();
  }

  llvm::LogicalResult mlirGenId(nlohmann::json &instrJson) {
    auto dest = instrJson["dest"].get<std::string>();
    auto argName = instrJson["args"][0].get<std::string>();

    if (!symbolTable.count(argName)) {
      llvm::errs() << "Undefined variable in id operation: " << argName << "\n";
      return mlir::failure();
    }

    auto arg = symbolTable.lookup(argName);

    if (llvm::failed(declare(dest, arg))) {
      llvm::errs() << "Failed to declare variable: " << dest << "\n";
      return mlir::failure();
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenNot(nlohmann::json &instrJson) {
    auto dest = instrJson["dest"].get<std::string>();
    auto argName = instrJson["args"][0].get<std::string>();

    if (!symbolTable.count(argName)) {
      llvm::errs() << "Undefined variable in not operation: " << argName
                   << "\n";
      return mlir::failure();
    }

    auto arg = symbolTable.lookup(argName);

    auto notOp = NotOp::create(builder, builder.getUnknownLoc(), arg);
    auto result = notOp.getResult();

    if (llvm::failed(declare(dest, result))) {
      llvm::errs() << "Failed to declare variable: " << dest << "\n";
      return mlir::failure();
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenBranch(nlohmann::json &instrJson) {
    auto argName = instrJson["args"][0].get<std::string>();

    if (!symbolTable.count(argName)) {
      llvm::errs() << "Undefined variable in branch operation: " << argName
                   << "\n";
      return mlir::failure();
    }

    auto arg = symbolTable.lookup(argName);

    auto trueLabel = instrJson["labels"][0].get<std::string>();
    auto falseLabel = instrJson["labels"][1].get<std::string>();

    if (!labelToBlock.count(trueLabel) || !labelToBlock.count(falseLabel)) {
      llvm::errs() << "Undefined label in branch operation: " << trueLabel
                   << " or " << falseLabel << "\n";
      return mlir::failure();
    }

    auto trueBlock = labelToBlock.lookup(trueLabel);
    auto falseBlock = labelToBlock.lookup(falseLabel);

    // TODO: get target block args from their get operations
    // auto brOp = BrOp::create(builder, builder.getUnknownLoc(), arg,
    // trueBlock,
    //                          falseBlock);

    return llvm::success();
  }

  llvm::LogicalResult mlirGenCall(nlohmann::json &instrJson) {
    auto funcName = instrJson["funcs"][0].get<std::string>();

    auto args = instrJson["args"];

    SmallVector<mlir::Value, 4> mlirArgs = {};
    for (auto &argNameJson : args) {
      auto argName = argNameJson.get<std::string>();
      if (!symbolTable.count(argName)) {
        llvm::errs() << "Undefined variable in call operation: " << argName
                     << "\n";
        return mlir::failure();
      }
      mlirArgs.push_back(symbolTable.lookup(argName));
    }

    auto type = getType(instrJson["type"].get<std::string>());

    if (instrJson.contains("dest")) {
      auto dest = instrJson["dest"].get<std::string>();
      auto callOp = CallOp::create(
          builder, builder.getUnknownLoc(), type,
          mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName),
          mlirArgs, mlir::ArrayAttr(), mlir::ArrayAttr());

      if (llvm::failed(declare(dest, callOp.getResult(0)))) {
        llvm::errs() << "Failed to declare variable: " << dest << "\n";
        return mlir::failure();
      }
    } else {
      CallOp::create(
          builder, builder.getUnknownLoc(), mlir::TypeRange{},
          mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName),
          mlirArgs, mlir::ArrayAttr(), mlir::ArrayAttr());
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenPrint(nlohmann::json &instrJson) {
    SmallVector<mlir::Value, 4> args = {};

    for (auto &argNameJson : instrJson["args"]) {
      auto argName = argNameJson.get<std::string>();

      if (!symbolTable.count(argName)) {
        llvm::errs() << "Undefined variable in print operation: " << argName
                     << "\n";
        return mlir::failure();
      }

      args.push_back(symbolTable.lookup(argName));
    }

    PrintOp::create(builder, builder.getUnknownLoc(), args);

    return llvm::success();
  }

  llvm::LogicalResult mlirGenNop(nlohmann::json &instrJson) {
    NopOp::create(builder, builder.getUnknownLoc());
    return llvm::success();
  }

  llvm::LogicalResult mlirGenRet(nlohmann::json &instrJson) {
    if (instrJson.contains("args")) {
      auto argName = instrJson["args"][0].get<std::string>();

      if (!symbolTable.count(argName)) {
        llvm::errs() << "Undefined variable in ret operation: " << argName
                     << "\n";
        return mlir::failure();
      }

      auto arg = symbolTable.lookup(argName);

      auto retOp = RetOp::create(builder, builder.getUnknownLoc(), arg);
    } else {
      // TODO: handle void return
      // auto retOp = RetOp::create(builder, builder.getUnknownLoc());
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenJmp(nlohmann::json &instrJson) {
    auto targetLabel = instrJson["labels"][0].get<std::string>();

    if (!labelToBlock.count(targetLabel)) {
      llvm::errs() << "Undefined label in jmp operation: " << targetLabel
                   << "\n";
      return mlir::failure();
    }

    auto targetBlock = labelToBlock.lookup(targetLabel);

    // TODO: get target block args from their get operations
    // auto jmpOp = JmpOp::create(builder, builder.getUnknownLoc(),

    return llvm::success();
  }

  llvm::LogicalResult mlirGenBinaryOp(nlohmann::json &instrJson) {
    auto dest = instrJson["dest"].get<std::string>();
    auto arg1Name = instrJson["args"][0].get<std::string>();
    auto arg2Name = instrJson["args"][1].get<std::string>();

    if (!symbolTable.count(arg1Name) || !symbolTable.count(arg2Name)) {
      llvm::errs() << "Undefined variable in binary operation: " << arg1Name
                   << " or " << arg2Name << "\n";
      return mlir::failure();
    }

    auto arg1 = symbolTable.lookup(arg1Name);
    auto arg2 = symbolTable.lookup(arg2Name);

    mlir::Value result;

    if (instrJson["op"] == "add") {
      auto addOp = AddOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = addOp.getResult();
    } else if (instrJson["op"] == "sub") {
      auto subOp = SubOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = subOp.getResult();
    } else if (instrJson["op"] == "mul") {
      auto mulOp = MulOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = mulOp.getResult();
    } else if (instrJson["op"] == "div") {
      auto divOp = DivOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = divOp.getResult();
    } else if (instrJson["op"] == "eq") {
      auto eqOp = EqOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = eqOp.getResult();
    } else if (instrJson["op"] == "lt") {
      auto ltOp = LtOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = ltOp.getResult();
    } else if (instrJson["op"] == "gt") {
      auto gtOp = GtOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = gtOp.getResult();
    } else if (instrJson["op"] == "le") {
      auto leOp = LeOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = leOp.getResult();
    } else if (instrJson["op"] == "ge") {
      auto geOp = GeOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = geOp.getResult();
    } else if (instrJson["op"] == "and") {
      auto andOp = AndOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = andOp.getResult();
    } else if (instrJson["op"] == "or") {
      auto orOp = OrOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = orOp.getResult();
    }

    if (llvm::failed(declare(dest, result))) {
      llvm::errs() << "Failed to declare variable: " << dest << "\n";
      return mlir::failure();
    }

    return llvm::success();
  }
};

} // namespace

namespace bril {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          nlohmann::json &json) {
  return MLIRGenImpl(context).mlirGen(json);
}

} // namespace bril
