import streams

type
  model_name* = enum
    cbow = 1, sg, sup


type
  loss_name*  = enum
    hs = 1, ns, softmax


type
  Args* = object
    input* : string
    output* : string
    lr* : cdouble
    lrUpdateRate* : cint
    dim* : cint
    ws* : cint
    epoch* : cint
    minCount* : cint
    minCountLabel* : cint
    neg* : cint
    wordNgrams* : cint
    loss* : loss_name
    model* : model_name
    bucket*: cint
    minn*: cint
    maxn* : cint
    thread* : cint
    t* : cdouble
    label* : string
    verbose* : cint
    pretrainedVectors* : string
    saveOutput* : bool
    qout* : bool
    retrain* : bool
    qnorm* : bool
    cutoff* : csize
    dsub* : csize

proc initArgs*(): Args =
    result.lr = 0.05
    result.dim = 100
    result.ws = 5
    result.epoch = 5
    result.minCount = 5
    result.minCountLabel = 0
    result.neg = 5
    result.wordNgrams = 1
    result.loss = loss_name.ns
    result.model = model_name.sg
    result.bucket = 2000000
    result.minn = 3
    result.maxn = 6
    result.thread = 12
    result.lrUpdateRate = 100
    result.t = 1e-4
    result.label = "__label__"
    result.verbose = 2
    result.pretrainedVectors = ""
    result.saveOutput = false

    result.qout = false
    result.retrain = false
    result.qnorm = false
    result.cutoff = 0
    result.dsub = 2

func lossToString( ln:loss_name):string =
  case (ln) :
    of loss_name.hs:
      return "hs";
    of loss_name.ns:
      return "ns";
    of loss_name.softmax:
      return "softmax";
  
  return "Unknown loss!"; # should never happen


func boolToString( b:bool):string =
  if (b):
    return "true";
  else:
    return "false";
  

func modelToString( mn:model_name):string = 
  case (mn) :
    of model_name.cbow:
      return "cbow";
    of model_name.sg:
      return "sg";
    of model_name.sup:
      return "sup";
  return "Unknown model name!"; # should never happen

# proc parseArgs*(this: var Args; args: vector[string]) {.stdcall,
#     importcpp: "parseArgs", header: headerargs.}
# proc printHelp*(this: var Args) {.stdcall, importcpp: "printHelp", header: headerargs.}
# proc printBasicHelp*(this: var Args) {.stdcall, importcpp: "printBasicHelp",
#                                    header: headerargs.}
# proc printDictionaryHelp*(this: var Args) {.stdcall,
#                                         importcpp: "printDictionaryHelp",
#                                         header: headerargs.}
# proc printTrainingHelp*(this: var Args) {.stdcall, importcpp: "printTrainingHelp",
#                                       header: headerargs.}
# proc printQuantizationHelp*(this: var Args) {.stdcall,
#     importcpp: "printQuantizationHelp", header: headerargs.}

{.this: self.}   
proc save*(self: var Args; a2: var Stream) =
    a2.writeData(addr dim,sizeof(dim))
    a2.writeData(addr ws,sizeof(ws))
    a2.writeData(addr epoch,sizeof(epoch))
    a2.writeData(addr minCount,sizeof(minCount))
    a2.writeData(addr neg,sizeof(neg))
    a2.writeData(addr wordNgrams,sizeof(wordNgrams))
    a2.writeData(addr loss,sizeof(loss))
    a2.writeData(addr model,sizeof(model))
    a2.writeData(addr bucket,sizeof(bucket))
    a2.writeData(addr minn,sizeof(minn))
    a2.writeData(addr maxn,sizeof(maxn))
    a2.writeData(addr lrUpdateRate,sizeof(lrUpdateRate))
    a2.writeData(addr t,sizeof(t))

{.this: self.}                                   
proc load*(self: var Args; a2: var Stream) =
    discard a2.readData(addr dim,sizeof(dim))
    discard a2.readData(addr ws,sizeof(ws))
    discard a2.readData(addr epoch,sizeof(epoch))
    discard a2.readData(addr minCount,sizeof(minCount))
    discard a2.readData(addr neg,sizeof(neg))
    discard a2.readData(addr wordNgrams,sizeof(wordNgrams))
    discard a2.readData(addr loss,sizeof(loss))
    discard a2.readData(addr model,sizeof(model))
    discard a2.readData(addr bucket,sizeof(bucket))
    discard a2.readData(addr minn,sizeof(minn))
    discard a2.readData(addr maxn,sizeof(maxn))
    discard a2.readData(addr lrUpdateRate,sizeof(lrUpdateRate))
    discard a2.readData(addr t,sizeof(t))

{.this: self.}  
proc dump*(self: Args; a2: var Stream) = 
    a2.writeLine "dim" & " " & $dim 
    a2.writeLine "ws" & " " & $ws 
    a2.writeLine "epoch" & " " & $epoch 
    a2.writeLine "minCount" & " " & $minCount 
    a2.writeLine "neg" & " " & $neg 
    a2.writeLine "wordNgrams" & " " & $wordNgrams 
    a2.writeLine "loss" & " " & lossToString(loss) 
    a2.writeLine "model" & " " & modelToString(model) 
    a2.writeLine "bucket" & " " & $bucket 
    a2.writeLine "minn" & " " & $minn 
    a2.writeLine "maxn" & " " & $maxn 
    a2.writeLine "lrUpdateRate" & " " & $lrUpdateRate 
    a2.writeLine "t" & " " & $t 