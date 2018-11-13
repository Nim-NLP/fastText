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



proc addRow*(self: var Matrix; vec: Vector; i: int64; a: float32) =
    doassert i >= 0
    doassert i < self.m
    doassert vec.size == self.n
    for j in countup(0,self.n.int32):
        self.idata[ (i * self.n + j).int32 ] += a * vec.get(j)

proc multiplyRow*(self: var Matrix; nums: Vector; ib: int64 = 0; ie: int64 = -1) =
    var iee = ie
    if ie == -1:
        iee = self.m
    doassert iee <= nums.size
    var i = ib
    var n:float32
    while i < iee:
        n = nums.get(i - ib)
        if n != 0:
            for j in countup(0'i64,self.n):
                self.at(i,j)[] *= n
        inc i


proc divideRow*(self: var Matrix; denoms: Vector; ib: int64 = 0; ie: int64 = -1) =
    var iee = ie
    if ie == -1:
        iee = self.m
    doassert iee <= denoms.size
    var i = ib
    var n:float32
    while i < iee:
        n = denoms.get(i - ib)
        if n != 0:
            for j in countup(0'i64,self.n):
                self.at(i,j)[] /= n
        inc i 

proc l2NormRow*(self: Matrix; i: int64): float32 {.noSideEffect.} = 
    var norm:float32 = 0.0
    for j in 0..<self.n:
        norm += self.at(i,j)
    
    if norm == NaN:
        raise newException(ValueError,"Encountered NaN.")
    sqrt(norm)

proc l2NormRow*(self: Matrix; norms: var Vector) {.noSideEffect.} =
    doassert norms.size == self.m
    for i in 0..<self.m:
        norms[i][] = self.l2NormRow(i)

proc save*(self: var Matrix; o: var Stream) =
    o.writeData(self.m.addr,sizeof(int64))
    o.writeData(self.n.addr,sizeof(int64))
    o.writeData(self.idata.addr, (int32) self.m * self.n * sizeof(float32))

type ssize_t = int32

proc load*(self: var Matrix; i: var Stream) = 
    discard i.readData(self.m.addr,sizeof(int64))
    discard i.readData(self.n.addr,sizeof(int64))
    debugEcho self.m,"m"
    debugEcho self.n,"n"
    
    self.idata.setLen(self.m * self.n)
    # discard i.readData(self.idata.addr,(int32) self.m * self.n * sizeof(float32))
    # debugEcho self.m * self.n,"matrix data size"
    for j in 0..<self.m * self.n :
        discard i.readData(self.idata[j].addr, sizeof(cfloat))

# proc dump*(self: Matrix; o: var Stream) {.noSideEffect.} =
