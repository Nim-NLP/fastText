
import streams
import ./types
import ./args
import ./matrix
import ./qmatrix
import ./dictionary
import ./model
import math

export dictionary
export args

const FASTTEXT_VERSION = 12'i32
const FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314'i32

type
    FastText* = object
        args*:ref Args
        dict*:ref Dictionary
        input: Matrix
        output: Matrix
        qinput: QMatrix
        qoutput: QMatrix
        model: Model

        quant: bool
        version: int32

proc initFastText*(): FastText =
    result.quant = false
    # result.version = FASTTEXT_VERSION

proc checkModel*(self: var FastText, i: var Stream): bool =
    var magic: int32
    discard i.readData(magic.addr, sizeof(int32))
    if magic != FASTTEXT_FILEFORMAT_MAGIC_INT32:
        return false
    discard i.readData(self.version.addr, sizeof(int32))
    if self.version > FASTTEXT_VERSION:
        return false
    return true

proc loadModel*(self: var FastText; i: var Stream) =
    self.args = newArgs()
    self.input = initMatrix()
    self.output = initMatrix()
    self.qinput = initQMatrix()
    self.qoutput = initQMatrix()
    self.args[].load(i)

    if self.version == 11 and self.args.model == model_name.sup:
        self.args.maxn = 0

    self.dict = newDictionary(self.args, i)
    var quant_input: bool
    discard i.readData(quant_input.addr, sizeof(bool))
    if quant_input:
        debugEcho "loadModel quant_input"
        self.quant = true
        self.qinput.load(i)
    else:
        debugEcho "loadModel input"
        self.input.load(i)
    if not quant_input and self.dict[].isPruned():
        raise newException(ValueError,
                """Invalid model file.\n
                  Please download the updated model from www.fasttext.cc.\n
                  See issue #332 on Github for more information.\n""")

    discard i.readData(self.args.qout.addr, sizeof(bool))
    if self.quant and self.args.qout:
        self.qoutput.load(i)
    else:
        self.output.load(i)
    self.model = initModel(self.input.addr,self.output.addr,self.args,0)
    self.model.quant = self.quant
    self.model.setQuantizePointer(self.qinput.addr,self.qoutput.addr,self.args.qout)
    if self.args.model == model_name.sup:
        self.model.setTargetCounts(self.dict[].getCounts(entry_type.label))
    else:
        self.model.setTargetCounts(self.dict[].getCounts(entry_type.word))
    debugEcho "load model end",self.args[]

proc loadModel*(self: var FastText; filename: string) =
    var ifs = openFileStream(filename)
    
    if not self.checkModel((Stream)ifs):
        raise newException(ValueError, (filename & " has wrong file format!"))
    self.loadModel((Stream)ifs)
    ifs.close()

proc predict*(self: var FastText; i:  Stream; k: int32;
        predictions: var seq[tuple[first: float32, second: string]];
        threshold: float32 = 0.0 ) =
    var words, labels: seq[int32]
    predictions.setLen(0)
    discard self.dict[].getLine(i, words, labels)
    predictions.setLen(0)
    if words.len == 0: return
    var hidden = initVector(self.args.dim)
    var output = initVector(self.dict.nlabels)
    var modelPredictions: seq[tuple[first: float32, second: int32]]
    debugEcho "words input len",words.len
    self.model.predict(words, k, threshold, modelPredictions, hidden.addr,
            output.addr)
    for it in modelPredictions:
        predictions.add( (first: exp(it.first),
                second: self.dict[].getLabel(it.second)))

proc predict*(self: var FastText; i:  Stream; k: int32; print_prob: bool;
        threshold: float32 = 0.0 ) =
    var line: string
    var predictions: seq[tuple[first: float32, second: string]]
    while i.readLine(line):
        predictions.setLen(0)
        self.predict(i, k, predictions, threshold)
        if predictions.len == 0:
            writeLine(stdout, "")
            continue
        for i in 0..<predictions.len:
            if i != 0:
                stdout.write(" ")
            stdout.write(predictions[i].second)
            if print_prob:
                stdout.write(" " & $exp(predictions[i].first))
        stdout.writeLine("")
    i.close()

# fasttext_pybind.cc interface
proc predict*(self: var FastText; text: string; k: int32 = 1;
        threshold: float32 = 0.0 ): seq[tuple[first: float32, second: string]] =
    var stream = (Stream)newStringStream(text)
    debugEcho "stream end"
    self.predict(stream,k,result,threshold)
