import random
import math
import streams
import ./types
import ./vector

proc initMatrix*(m: int64; n: int64): Matrix =
    result.idata.setLen(m*n)
    result.m = m
    result.n = n

proc initMatrix*(): Matrix =
    result

proc zero*(self: var Matrix) =
    for i in 0..<self.idata.len:
        self.idata[i] = 0.0

proc uniform*(self: var Matrix; a: float32) =
    var rng = initRand(1)
    for i in 0..<(self.m * self.n):
        self.idata[i.int32] = rng.rand( -a..a)

proc save*(self: var Matrix; o: var Stream) =
    o.writeData(self.m.addr, sizeof(int64))
    o.writeData(self.n.addr, sizeof(int64))
    o.writeData(self.idata.addr, (int32)self.m * self.n * sizeof(float32))

type ssize_t = int32

proc load*(self: var Matrix; i: var Stream) =
    discard i.readData(self.m.addr, sizeof(int64))
    discard i.readData(self.n.addr, sizeof(int64))
    self.idata.setLen(self.m * self.n)
    for j in 0..<self.m * self.n:
        discard i.readData(self.idata[j].addr, sizeof(float32))

# proc dump*(self: Matrix; o: var Stream) {.noSideEffect.} =
