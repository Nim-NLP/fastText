
import streams
import ./args
import ./matrix
import ./qmatrix
import ./dictionary
import math

const  FASTTEXT_VERSION = 12'i32
const  FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314'i32

type
    FastText* = object
        quant:bool
        version:int32
        dict:ptr Dictionary

proc initFastText*(): FastText =
    result.quant = false
    # result.version

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
                  See issue #332 on Github for more information.\n""")
   
    discard i.readData(args.qout.addr,sizeof(bool))
    if self.quant and args.qout:
        qoutput.load(i)
    else:
        output.load(i)

proc loadModel*(self: var FastText; filename: string) =
    var ifs = openFileStream(filename)
    self.loadModel((Stream)ifs)
    if not self.checkModel((Stream)ifs):
        raise newException(ValueError,(filename & " has wrong file format!"))
    ifs.close()

proc predict*(self:var  FastText; i: var Stream; k: int32;predictions: var seq[tuple[first:float32, second:string]]; threshold: float32 = 0.0) {.noSideEffect.} =
    var words,labels:seq[int32]
    predictions.setLen(0)
    self.dict.getLine(i,words,labels)
    
proc predict*(self: var FastText; i: var Stream; k: int32; print_prob: bool; threshold: float32 = 0.0) =
    var line:string
    var predictions:seq[tuple[first:float32,second:string]]
    while i.readLine(line):
        predictions.setLen(0)
        self.predict(i,k,predictions,threshold)
        if predictions.len == 0:
            writeLine(stdout,"")
            continue
        for i in 0..<predictions.len:
            if i != 0:
                stdout.write(" ")
            stdout.write(predictions[i].second)
            if print_prob:
                stdout.write(" " & $exp(predictions[i].first))
        stdout.writeLine("")
    i.close()

