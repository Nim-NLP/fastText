import random
import math
import streams
import ./types
import ./vector

proc initMatrix*(m: int64; n: int64): Matrix =
    result = Matrix(idata:newSeq[float32](m * n),m:m,n:n)
    
proc initMatrix*(): Matrix =
    result = initMatrix(0,0)

proc initMatrix*(a1: Matrix): Matrix =
    result = a1


proc zero*(self: var Matrix) =
    for i in countup(0,self.idata.len):
        self.idata[i] = 0.0

proc uniform*(self: var Matrix; a: float32) =
    randomize(1)
    for i in countup(0'i64, (self.m * self.n) ):
        self.idata[i.int32] = rand( -a..a)

proc save*(self: var Matrix; o: var Stream) =
    o.writeData(self.m.addr,sizeof(int64))
    o.writeData(self.n.addr,sizeof(int64))
    o.writeData(self.idata.addr, (int32) self.m * self.n * sizeof(float32))

type ssize_t = int32

proc load*(self: var Matrix; i: var Stream) = 
    discard i.readData(self.m.addr,sizeof(int64))
    discard i.readData(self.n.addr,sizeof(int64))
    self.idata.setLen(self.m * self.n)
    for j in 0..<self.m * self.n :
        discard i.readData(self.idata[j].addr, sizeof(cfloat))

# proc dump*(self: Matrix; o: var Stream) {.noSideEffect.} =
