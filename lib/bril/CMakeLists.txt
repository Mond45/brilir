add_mlir_dialect_library(MLIRBril
        BrilOps.cpp
        BrilDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/bril

        DEPENDS
        MLIRBrilOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
