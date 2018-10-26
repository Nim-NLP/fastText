
import streams
import ./args
import ./matrix
import ./qmatrix
import ./dictionary

const  FASTTEXT_VERSION = 12'i32
const  FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314'i32


type
    FastText* = object
        quant:bool
        version:int32

proc initFastText*(): FastText =
    result.quant = false
    # result.version

# proc addInputVector()
# # proc constructFastText*(): FastText {.stdcall, constructor,
# #                                    importcpp: "fasttext::FastText(@)",
# #                                    header: headerfasttext.}
# proc getWordId*(this: FastText; a2: std_string): int32 {.noSideEffect, stdcall,
#     importcpp: "getWordId", header: headerfasttext.}
# proc getSubwordId*(this: FastText; a2: std_string): int32 {.noSideEffect, stdcall,
#     importcpp: "getSubwordId", header: headerfasttext.}
# proc getVector*(this: FastText; a2: var Vector; a3: std_string) {.noSideEffect, stdcall,
#     importcpp: "getVector", header: headerfasttext.}
# proc getWordVector*(this: FastText; a2: var Vector; a3: std_string) {.noSideEffect, stdcall,
#     importcpp: "getWordVector", header: headerfasttext.}
# proc getSubwordVector*(this: FastText; a2: var Vector; a3: std_string) {.noSideEffect,
#     stdcall, importcpp: "getSubwordVector", header: headerfasttext.}
# proc addInputVector*(this: FastText; a2: var Vector; a3: int32) {.noSideEffect, stdcall,
#     importcpp: "addInputVector", header: headerfasttext.}
# proc getInputVector*(this: var FastText; vec: var Vector; ind: int32) {.stdcall,
#     importcpp: "getInputVector", header: headerfasttext.}
# proc getArgs*(this: FastText): Args {.noSideEffect, stdcall, importcpp: "getArgs",
#                                   header: headerfasttext.}
# proc getDictionary*(this: FastText): shared_ptr[Dictionary] {.noSideEffect, stdcall,
#     importcpp: "getDictionary", header: headerfasttext.}
# proc getInputMatrix*(this: FastText): shared_ptr[Matrix] {.noSideEffect, stdcall,
#     importcpp: "getInputMatrix", header: headerfasttext.}
# proc getOutputMatrix*(this: FastText): shared_ptr[Matrix] {.noSideEffect, stdcall,
#     importcpp: "getOutputMatrix", header: headerfasttext.}
# proc saveVectors*(this: var FastText) {.stdcall, importcpp: "saveVectors",
#                                     header: headerfasttext.}
# proc saveModel*(this: var FastText; a2: std_string) {.stdcall, importcpp: "saveModel",
#     header: headerfasttext.}
# proc saveOutput*(this: var FastText) {.stdcall, importcpp: "saveOutput",
#                                    header: headerfasttext.}
# proc saveModel*(this: var FastText) {.stdcall, importcpp: "saveModel",
#                                   header: headerfasttext.}
proc checkModel*(self:var FastText,i:var Stream):bool =
    var magic:int32
    discard i.readData(magic.addr,sizeof(int32))
    if magic != FASTTEXT_FILEFORMAT_MAGIC_INT32:
        return false
    discard i.readData(self.version.addr,sizeof(int32))
    if self.version > FASTTEXT_VERSION:
        return false
    return true

proc loadModel*(self: var FastText; i: var Stream) =
    var 
        args = initArgs()
        input = constructMatrix()
        output = constructMatrix()
        qinput = initQMatrix()
        qoutput = initQMatrix()
    args.load(i)
    if self.version == 11 and args.model == model_name.sup:
        args.maxn = 0
    var dict = initDictionary(args.addr,i)
    var quant_input:bool
    discard i.readData(quant_input.addr,sizeof(bool))
    if not quant_input and dict.isPruned():
        raise newException(ValueError,"""Invalid model file.\n
                  Please download the updated model from www.fasttext.cc.\n
                  See issue #332 on Github for more information.\n"""))
   
    discard i.readData(args.qout.addr,sizeof(bool))
    if self.quant and args.qout:
        qoutput.load(i)
    else:
        output.load(i)

proc loadModel*(self: var FastText; filename: string) =
    let ifs = openFileStream(filename)
    self.loadModel(ifs)
    if not checkModel(ifs):
        raise newException(ValueError,(filename & " has wrong file format!")
    
    ifs.close()
proc printInfo*(this: var FastText; a2: real; a3: real; a4: var ostream) {.stdcall,
    importcpp: "printInfo", header: headerfasttext.}
proc supervised*(this: var FastText; a2: var Model; a3: real; a4: vect[int32];
                a5: vect[int32]) {.stdcall, importcpp: "supervised",
                                   header: headerfasttext.}
proc cbow*(this: var FastText; a2: var Model; a3: real; a4: vect[int32]) {.stdcall,
    importcpp: "cbow", header: headerfasttext.}
proc skipgram*(this: var FastText; a2: var Model; a3: real; a4: vect[int32]) {.stdcall,
    importcpp: "skipgram", header: headerfasttext.}
proc selectEmbeddings*(this: FastText; a2: int32): vect[int32] {.noSideEffect,
    stdcall, importcpp: "selectEmbeddings", header: headerfasttext.}
proc getSentenceVector*(this: var FastText; a2: var istream; a3: var Vector) {.stdcall,
    importcpp: "getSentenceVector", header: headerfasttext.}
proc quantize*(this: var FastText; a2: Args) {.stdcall, importcpp: "quantize",
    header: headerfasttext.}
proc test*(this: var FastText; a2: var istream; a3: int32; a4: real = 0.0): `tuple`[int64,
    cdouble, cdouble] {.stdcall, importcpp: "test", header: headerfasttext.}
# proc predict*(self: var FastText; i: var Stream; k: int32; print_prob: bool; threshold: float32 = 0.0) =
#     var line:string

#     while i.readLine(line):
#         echo line
#       i.close()
proc predict*(self:var  FastText; i: var Stream; k: int32;predictions: var seq[tuple[first:float32, second:string]]; threshold: float32 = 0.0) {.noSideEffect.} =
    predictions.resize(0)
    # self.dict
proc ngramVectors*(this: var FastText; a2: std_string) {.stdcall,
    importcpp: "ngramVectors", header: headerfasttext.}
proc precomputeWordVectors*(this: var FastText; a2: var Matrix) {.stdcall,
    importcpp: "precomputeWordVectors", header: headerfasttext.}
proc findNN*(this: var FastText; a2: Matrix; a3: Vector; a4: int32; a5: std_set[std_string];
            results: var vect[pair[real, std_string]]) {.stdcall, importcpp: "findNN",
    header: headerfasttext.}
proc analogies*(this: var FastText; a2: int32) {.stdcall, importcpp: "analogies",
    header: headerfasttext.}
proc trainThread*(this: var FastText; a2: int32) {.stdcall, importcpp: "trainThread",
    header: headerfasttext.}
proc train*(this: var FastText; a2: Args) {.stdcall, importcpp: "train",
                                      header: headerfasttext.}
proc loadVectors*(this: var FastText; a2: std_string) {.stdcall, importcpp: "loadVectors",
    header: headerfasttext.}
proc getDimension*(this: FastText): cint {.noSideEffect, stdcall,
                                       importcpp: "getDimension",
                                       header: headerfasttext.}
proc isQuant*(this: FastText): bool {.noSideEffect, stdcall, importcpp: "isQuant",
                                  header: headerfasttext.}