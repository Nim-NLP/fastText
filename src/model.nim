type
  shared_ptr {.importc: "std::shared_ptr", header: "<memory>".}[T] = object
  minstd_rand {.importc: "std::minstd_rand", header: "<random>".} = object
  vect {.importc: "std::vector", header: "<vector>".}[T] = object
  pair {.importc: "std::pair", header: "<utility>".}[T, U] = object
  real = float
import nimfastText/args
import nimfastText/matrix
import nimfastText/vector
import nimfastText/qmatrix
#import nimfastText/real
import strutils
const sourcePath = currentSourcePath().split({'\\', '/'})[0..^2].join("/")
{.passC: "-I\"" & sourcePath & "/src\"".}
const headermodel = sourcePath & "/src/model.h"
type
  Node* {.importcpp: "fasttext::Node", header: headermodel, bycopy.} = object
    parent* {.importc: "parent".}: int32
    left* {.importc: "left".}: int32
    right* {.importc: "right".}: int32
    count* {.importc: "count".}: int64
    binary* {.importc: "binary".}: bool

  Model* {.importcpp: "fasttext::Model", header: headermodel, bycopy.} = object
    rng* {.importc: "rng".}: minstd_rand
    quant* {.importc: "quant_".}: bool


proc constructModel*(a1: shared_ptr[Matrix]; a2: shared_ptr[Matrix];
                    a3: shared_ptr[Args]; a4: int32): Model {.stdcall, constructor,
    importcpp: "fasttext::Model(@)", header: headermodel.}
proc binaryLogistic*(this: var Model; a2: int32; a3: bool; a4: real): real {.stdcall,
    importcpp: "binaryLogistic", header: headermodel.}
proc negativeSampling*(this: var Model; a2: int32; a3: real): real {.stdcall,
    importcpp: "negativeSampling", header: headermodel.}
proc hierarchicalSoftmax*(this: var Model; a2: int32; a3: real): real {.stdcall,
    importcpp: "hierarchicalSoftmax", header: headermodel.}
proc softmax*(this: var Model; a2: int32; a3: real): real {.stdcall, importcpp: "softmax",
    header: headermodel.}
proc predict*(this: Model; a2: vect[int32]; a3: int32; a4: real;
             a5: var vect[pair[real, int32]]; a6: var Vector; a7: var Vector) {.
    noSideEffect, stdcall, importcpp: "predict", header: headermodel.}
proc predict*(this: var Model; a2: vect[int32]; a3: int32; a4: real;
             a5: var vect[pair[real, int32]]) {.stdcall, importcpp: "predict",
    header: headermodel.}
proc dfs*(this: Model; a2: int32; a3: real; a4: int32; a5: real;
         a6: var vect[pair[real, int32]]; a7: var Vector) {.noSideEffect, stdcall,
    importcpp: "dfs", header: headermodel.}
proc findKBest*(this: Model; a2: int32; a3: real; a4: var vect[pair[real, int32]];
               a5: var Vector; a6: var Vector) {.noSideEffect, stdcall,
    importcpp: "findKBest", header: headermodel.}
proc update*(this: var Model; a2: vect[int32]; a3: int32; a4: real) {.stdcall,
    importcpp: "update", header: headermodel.}
proc computeHidden*(this: Model; a2: vect[int32]; a3: var Vector) {.noSideEffect,
    stdcall, importcpp: "computeHidden", header: headermodel.}
proc computeOutputSoftmax*(this: Model; a2: var Vector; a3: var Vector) {.noSideEffect,
    stdcall, importcpp: "computeOutputSoftmax", header: headermodel.}
proc computeOutputSoftmax*(this: var Model) {.stdcall,
    importcpp: "computeOutputSoftmax", header: headermodel.}
proc setTargetCounts*(this: var Model; a2: vect[int64]) {.stdcall,
    importcpp: "setTargetCounts", header: headermodel.}
proc initTableNegatives*(this: var Model; a2: vect[int64]) {.stdcall,
    importcpp: "initTableNegatives", header: headermodel.}
proc buildTree*(this: var Model; a2: vect[int64]) {.stdcall, importcpp: "buildTree",
    header: headermodel.}
proc getLoss*(this: Model): real {.noSideEffect, stdcall, importcpp: "getLoss",
                               header: headermodel.}
proc sigmoid*(this: Model; a2: real): real {.noSideEffect, stdcall,
                                       importcpp: "sigmoid", header: headermodel.}
proc log*(this: Model; a2: real): real {.noSideEffect, stdcall, importcpp: "log",
                                   header: headermodel.}
proc std_log*(this: Model; a2: real): real {.noSideEffect, stdcall,
                                       importcpp: "std_log", header: headermodel.}
proc setQuantizePointer*(this: var Model; a2: shared_ptr[QMatrix];
                        a3: shared_ptr[QMatrix]; a4: bool) {.stdcall,
    importcpp: "setQuantizePointer", header: headermodel.}