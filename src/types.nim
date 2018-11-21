import math
import random

type
    ProductQuantizer*  = object
      nbits*:int32
      ksub*:int32
      max_points*:int32
      eps*:float32
      niter*:int32
      max_points_per_cluster*:int32
      dim*,nsubq*,dsub*,lastdsub*:int32
      centroids*: seq[float32]
      seed*:int32
      rng*: Rand

type
    Matrix* = object
      idata*:seq[float32]
      m*,n*:int64
    QMatrix*  = object
      qnorm*:bool
      m*,n*:int64
      codesize*:int32
      codes*:seq[uint8]
      norm_codes*:seq[uint8]
      pq*,npq*:ref ProductQuantizer

proc size*(self: Matrix; dim: int64): int64 =
    assert(dim == 0 or dim == 1 )
    result = if dim == 0 : self.m else : self.n

type
    Vector*  = object
        idata*:seq[ float32]

proc initVector*(a1: int64): Vector =
    result.idata = newSeq[float32](a1)

proc initVector*(a1: Vector): Vector =
    result = a1
        

proc size*(self: Vector): int64 =
    self.idata.len

proc data*(self: var Vector): ptr seq[float32] =
    addr self.idata

proc data*(self: Vector): ptr seq[float32] =
    self.idata.unSafeAddr

proc `[]`*(self: var Vector; i: int64): ptr float32 =
    result = self.idata[i].addr

proc `[]`*(self:  Vector; i: int64): ptr float32 =
    result = self.idata[i].unsafeAddr
# proc `[]=`*(self: var Vector; i: int64,j:float32)  =
#     self.idata[i] = j

# proc `[]+=`*(self: var Vector; i: int64,j:float32)  =
#     self.idata[i] = self.idata[i] + j
    
proc get*(self: Vector; i: int64): float32 =
    self.idata[i]

# proc data*(self: var Matrix): ptr float32 =
#     self.idata[0].addr

# proc data*(self: Matrix): ptr float32 {.noSideEffect.} =
#     self.idata[0].unsafeAddr

proc at*(self: Matrix; i: int64; j: int64): float32 {.noSideEffect.} =
    self.idata[ (i * self.n + j) ]

proc at*(self: var Matrix; i: int64; j: int64): ptr float32 =
    self.idata[ (i * self.n + j) ].unsafeAddr

proc rows*(self: Matrix): int64 =
    self.m

proc cols*(self: Matrix): int64 = 
    self.n

proc dotRow*(self: Matrix; vec: Vector; i: int32): float32 {.noSideEffect.} =
    assert i >= 0
    assert i < self.m
    assert vec.size == self.n
    for j in 0..<self.n:
        result += self.at(i,j) * vec.get(j.int64)
    if classify(result) == math.fcNan:
        raise newException(ValueError,"Encountered NaN.")

proc getM*(self: QMatrix): int64 =
    self.m

proc getN*(self: QMatrix): int64 =
    self.n

const nbits:int32 = 8;
const ksub*:int32 = 1 shl nbits;
const max_points_per_cluster:int32 = 256;
const max_points:int32 = max_points_per_cluster * ksub
const seed:int32 = 1234;
const niter:int32 = 25;
const eps:float32 = 1e-7.float32;

# proc `[]`(self:var uint8,key:int):uint8 = 
#     let a: UncheckedArray[uint8] = cast[ UncheckedArray[uint8]](self)
#     (uint8)a[key]

proc getCentroidsPosition*(self:  ProductQuantizer; m: int32; i: uint8): int32 =
    if (m == self.nsubq - 1) :
        return m * ksub * self.dsub + i.int32 * self.lastdsub
    return (m * ksub + i.int32) * self.dsub
    

proc mulcode*(self: ProductQuantizer; x:var Vector; codes: seq[uint8];codePos:int32; t: int32; alpha: float32): float32 =
    var res = 0.0'f32
    var d = self.dsub
    var codePos:int32 = codePos + self.nsubq * t
    var cp:int32
    for m in 0..<self.nsubq:
        cp = self.getCentroidsPosition(m.int32,codes[codePos+m])
        if m == self.nsubq - 1 :
            d = self.lastdsub
        for n in 0..<d:
            res += x[int64(m * self.dsub + n)][] * self.centroids[cp*n]
    result = res * alpha

proc addcode*(self:  ProductQuantizer; x: var Vector; codes: seq[uint8];codePos:int32; t: int32; alpha: float32) =
    var d = self.dsub
    var codePos:int32 = codePos + self.nsubq * t
    var cp:int32
    for m in 0..<self.nsubq:
        cp = self.getCentroidsPosition(m.int32,codes[m + codePos])
        if m == self.nsubq - 1 :
            d = self.lastdsub
        for n in 0..<d:
            x[m * self.dsub + n][] += (alpha * self.centroids[cp+n])

proc addToVector*(self: QMatrix; x: var Vector; t: int32) =
    var norm:float32 = 1
    var normPos:int32
    if self.qnorm:
        normPos = self.npq[].getCentroidsPosition(0'i32, self.norm_codes[t])
        norm = self.npq.centroids[normPos]
    self.pq[].addcode(x,self.codes, normPos, t, norm)

proc dotRow*(self: QMatrix; vec:var Vector; i: int64): float32 =
    assert(i >= 0);
    assert(i < self.m)
    assert(vec.size() == self.n)
    var norm:float32 = 1
    var normPos:int32
    if self.qnorm:
        debugEcho "getCentroidsPosition start"
        normPos = self.npq[].getCentroidsPosition(0'i32, self.norm_codes[i])
        debugEcho "getCentroidsPosition end",normPos
        norm = self.npq.centroids[normPos]
    debugEcho "mulcode"
    self.pq[].mulcode(vec,self.codes, normPos, i.int32, norm)

proc l2NormRow*(self:var Matrix; i: int64): float32 {.noSideEffect.} = 
    var norm:float32 = 0.0
    for j in 0..<self.n:
        norm += self.at(i,j)[]
    
    if norm == NaN:
        raise newException(ValueError,"Encountered NaN.")
    sqrt(norm)

proc l2NormRow*(self:var Matrix; norms: var Vector) {.noSideEffect.} =
    doassert norms.size == self.m
    for i in 0..<self.m:
        norms[i][] = self.l2NormRow(i)

proc addRow*(self: var Matrix; vec: Vector; i: int64; a: float32) =
    doassert i >= 0
    doassert i < self.m
    doassert vec.size == self.n
    for j in 0..<self.n:
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
            for j in 0..<self.n:
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
            for j in 0..<self.n:
                self.at(i,j)[] /= n
        inc i 


    

        
    