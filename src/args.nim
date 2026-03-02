import streams
import ./types


func lossToString(ln: loss_name): string =
  case (ln):
    of loss_name.hs:
      return "hs";
    of loss_name.ns:
      return "ns";
    of loss_name.softmax:
      return "softmax";


func modelToString(mn: model_name): string =
  case (mn):
    of model_name.cbow:
      return "cbow"
    of model_name.sg:
      return "sg"
    of model_name.sup:
      return "sup"

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

proc save*(self: var Args; a2: var Stream) =
  a2.writeData(addr self.dim, sizeof(int32))
  a2.writeData(addr self.ws, sizeof(int32))
  a2.writeData(addr self.epoch, sizeof(int32))
  a2.writeData(addr self.minCount, sizeof(int32))
  a2.writeData(addr self.neg, sizeof(int32))
  a2.writeData(addr self.wordNgrams, sizeof(int32))
  a2.writeData(addr self.loss, sizeof(int32))
  a2.writeData(addr self.model, sizeof(int32))
  a2.writeData(addr self.bucket, sizeof(int32))
  a2.writeData(addr self.minn, sizeof(int32))
  a2.writeData(addr self.maxn, sizeof(int32))
  a2.writeData(addr self.lrUpdateRate, sizeof(int32))
  a2.writeData(addr self.t, sizeof(float64))


proc load*(self: var Args; a2: var Stream) =
  discard a2.readData(addr self.dim, sizeof(int32))
  discard a2.readData(addr self.ws, sizeof(int32))
  discard a2.readData(addr self.epoch, sizeof(int32))
  discard a2.readData(addr self.minCount, sizeof(int32))
  discard a2.readData(addr self.neg, sizeof(int32))
  discard a2.readData(addr self.wordNgrams, sizeof(int32))
  discard a2.readData(addr self.loss, sizeof(int32))
  discard a2.readData(addr self.model, sizeof(int32))
  discard a2.readData(addr self.bucket, sizeof(int32))
  discard a2.readData(addr self.minn, sizeof(int32))
  discard a2.readData(addr self.maxn, sizeof(int32))
  discard a2.readData(addr self.lrUpdateRate, sizeof(int32))
  discard a2.readData(addr self.t, sizeof(float64))


proc dump*(self: Args; a2: var Stream) =
  a2.writeLine "dim" & " " & $self.dim
  a2.writeLine "ws" & " " & $self.ws
  a2.writeLine "epoch" & " " & $self.epoch
  a2.writeLine "minCount" & " " & $self.minCount
  a2.writeLine "neg" & " " & $self.neg
  a2.writeLine "wordNgrams" & " " & $self.wordNgrams
  a2.writeLine "loss" & " " & lossToString(self.loss)
  a2.writeLine "model" & " " & modelToString(self.model)
  a2.writeLine "bucket" & " " & $self.bucket
  a2.writeLine "minn" & " " & $self.minn
  a2.writeLine "maxn" & " " & $self.maxn
  a2.writeLine "lrUpdateRate" & " " & $self.lrUpdateRate
  a2.writeLine "t" & " " & $self.t
