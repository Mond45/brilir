//===- BrilOps.td - Bril dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BRIL_OPS
#define BRIL_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "bril/BrilDialect.td"

def ValidBrilType : AnyTypeOf<[SI32, I1]>;

def ConstantOp : Bril_Op<"const", [Pure]> {
    let summary = "Defines a constant value.";
    let description = [{
        The `bril.const` operation defines a constant integer or boolean value.
    }];

    let arguments = (ins AnyAttrOf<[SI32Attr, I1Attr]>:$value);
    let results = (outs ValidBrilType);

    let builders = [OpBuilder<(ins "int":$value)>, OpBuilder<(ins "bool":$value)>];
}

def AddOp : Bril_Op<"add", [Pure, SameOperandsAndResultType]> {
    let summary = "Adds two integers.";
    let description = [{
        The `bril.add` operation takes two integer operands and returns their sum.
    }];

    let arguments = (ins SI32:$lhs, SI32:$rhs);
    let results = (outs SI32);
}

def SubOp : Bril_Op<"sub", [Pure, SameOperandsAndResultType]> {
    let summary = "Subtracts two integers.";
    let description = [{
        The `bril.sub` operation takes two integer operands and returns the result of subtracting the second from the first.
    }];

    let arguments = (ins SI32:$lhs, SI32:$rhs);
    let results = (outs SI32);
}

def MulOp : Bril_Op<"mul", [Pure, SameOperandsAndResultType]> {
    let summary = "Multiplies two integers.";
    let description = [{
        The `bril.mul` operation takes two integer operands and returns their product.
    }];

    let arguments = (ins SI32:$lhs, SI32:$rhs);
    let results = (outs SI32);
}

def DivOp : Bril_Op<"div", [SameOperandsAndResultType]> {
    let summary = "Divides two integers.";
    let description = [{
        The `bril.div` operation takes two integer operands and returns the result of dividing the first by the second.
    }];

    let arguments = (ins SI32:$lhs, SI32:$rhs);
    let results = (outs SI32);
}

def EqOp : Bril_Op<"eq", [Pure]> {
    let summary = "Compares two integers for equality.";
    let description = [{
        The `bril.eq` operation takes two integer operands and returns true if they are equal, otherwise returns false.
    }];

    let arguments = (ins SI32:$lhs, SI32:$rhs);
    let results = (outs I1);
}

def LtOp : Bril_Op<"lt", [Pure]> {
    let summary = "Compares two integers for less-than.";
    let description = [{
        The `bril.lt` operation takes two integer operands and returns true if the first is less than the second, otherwise returns false.
    }];

    let arguments = (ins SI32:$lhs, SI32:$rhs);
    let results = (outs I1);
}

def GtOp : Bril_Op<"gt", [Pure]> {
    let summary = "Compares two integers for greater-than.";
    let description = [{
        The `bril.gt` operation takes two integer operands and returns true if the first is greater than the second, otherwise returns false.
    }];

    let arguments = (ins SI32:$lhs, SI32:$rhs);
    let results = (outs I1);
}

def LeOp : Bril_Op<"le", [Pure]> {
    let summary = "Compares two integers for less-than-or-equal.";
    let description = [{
        The `bril.le` operation takes two integer operands and returns true if the first is less than or equal to the second, otherwise returns false.
    }];

    let arguments = (ins SI32:$lhs, SI32:$rhs);
    let results = (outs I1);
}

def GeOp : Bril_Op<"ge", [Pure]> {
    let summary = "Compares two integers for greater-than-or-equal.";
    let description = [{
        The `bril.ge` operation takes two integer operands and returns true if the first is greater than or equal to the second, otherwise returns false.
    }];

    let arguments = (ins SI32:$lhs, SI32:$rhs);
    let results = (outs I1);
}

def NotOp : Bril_Op<"not", [Pure]> {
    let summary = "Logical NOT operation.";
    let description = [{
        The `bril.not` operation takes a boolean operand and returns its logical negation.
    }];

    let arguments = (ins I1:$operand);
    let results = (outs I1);
}

def AndOp : Bril_Op<"and", [Pure]> {
    let summary = "Logical AND operation.";
    let description = [{
        The `bril.and` operation takes two boolean operands and returns true if both are true, otherwise returns false.
    }];

    let arguments = (ins I1:$lhs, I1:$rhs);
    let results = (outs I1);
}

def OrOp : Bril_Op<"or", [Pure]> {
    let summary = "Logical OR operation.";
    let description = [{
        The `bril.or` operation takes two boolean operands and returns true if at least one is true, otherwise returns false.
    }];

    let arguments = (ins I1:$lhs, I1:$rhs);
    let results = (outs I1);
}

def CallOp : Bril_Op<"call"> {
    let summary = "Calls a function.";
    let description = [{
        The `bril.call` operation invokes a function with the given arguments and returns the result.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<ValidBrilType>:$inputs,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let results = (outs Optional<ValidBrilType>);

    let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def IdOp : Bril_Op<"id", [Pure]> {
    let summary = "Identity operation.";
    let description = [{
        The `bril.id` operation returns its input operand unchanged.
    }];

    let arguments = (ins ValidBrilType:$input);
    let results = (outs ValidBrilType);
}

def JmpOp : Bril_Op<"jmp", [Pure, Terminator]> {
    let summary = "Unconditional jump to a label.";
    let description = [{
        The `bril.jmp` operation transfers control unconditionally to the specified label.
    }];

    let arguments = (ins Variadic<ValidBrilType>:$args);
    let successors = (successor AnySuccessor:$target);

    let assemblyFormat = [{
    $target (`(` $args^ `:` type($args) `)`)? attr-dict
  }];
}

def BrOp: Bril_Op<"br", [Pure, Terminator, AttrSizedOperandSegments]> {
    let summary = "Conditional branch to labels.";
    let description = [{
        The `bril.br` operation transfers control to one of two specified labels based on a boolean condition.
    }];

    let arguments = (ins I1:$condition, Variadic<ValidBrilType>:$true_args, Variadic<ValidBrilType>:$false_args);
    let successors = (successor AnySuccessor:$true_target, AnySuccessor:$false_target);

    let assemblyFormat = [{
    $condition `,`
    $true_target (`(` $true_args^ `:` type($true_args) `)`)? `,`
    $false_target (`(` $false_args^ `:` type($false_args) `)`)?
    attr-dict
  }];
}

def FuncOp : Bril_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {
    let summary = "Defines a function.";
    let description = [{
        The `bril.func` operation defines a function with a name, arguments, return type, and body.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let regions = (region AnyRegion:$body);

    let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region on the function operation that is callable.
    Region *getCallableRegion() { return &getBody(); }
  }];

    let builders = [OpBuilder<(ins "StringRef":$name, "FunctionType":$type, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];
    let skipDefaultBuilders = 1;
}

def RetOp : Bril_Op<"ret", [Pure, Terminator, HasParent<"FuncOp">]> {
    let summary = "Return from a function.";
    let description = [{
        The `bril.ret` operation returns control from a function, optionally returning a value.
    }];

    let arguments = (ins Optional<ValidBrilType>:$return_value);
}

def PrintOp : Bril_Op<"print"> {
    let summary = "Prints a value.";
    let description = [{
        The `bril.print` operation outputs the given value to the standard output.
    }];

    let arguments = (ins Variadic<ValidBrilType>:$values);
}

def NopOp : Bril_Op<"nop"> {
    let summary = "No operation.";
    let description = [{
        The `bril.nop` operation performs no action.
    }];
}


#endif // BRIL_OPS
