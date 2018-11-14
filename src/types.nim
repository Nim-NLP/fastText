import ./productquantizer
type
    ProductQuantizer*  = object
      dim*,nsubq*,dsub*,lastdsub*:int32
      centroids*:ref seq[ ptr float32]
      # std::minstd_rand rng;
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
      pq*,npq*:ptr ProductQuantizer

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
    let a:ptr UncheckedArray[float32] = cast[ptr UncheckedArray[float32]](self.data)
    result = a[i.int32].addr

# proc `[]=`*(self: var Vector; i: int64,j:float32)  =
#     self.idata[i] = j

# proc `[]+=`*(self: var Vector; i: int64,j:float32)  =
#     self.idata[i] = self.idata[i] + j
    
proc get*(self: Vector; i: int64): float32 =
    self.idata[i.int32]

proc data*(self: var Matrix): ptr float32 =
    self.idata[0].addr

proc data*(self: Matrix): ptr float32 {.noSideEffect.} =
    self.idata[0].unsafeAddr

proc at*(self: Matrix; i: int64; j: int64): float32 {.noSideEffect.} =
    self.idata[ (i * self.n + j).int32 ]

proc at*(self: var Matrix; i: int64; j: int64): ptr float32 =
    self.idata[ (i * self.n + j).int32 ].unsafeAddr

proc rows*(self: Matrix): int64 =
    self.m

proc cols*(self: Matrix): int64 = 
    self.n

proc dotRow*(self: Matrix; vec: Vector; i: int64): float32 {.noSideEffect.} =
    doassert i >= 0
    doassert i < self.m
    doassert vec.size == self.n
    var d:float32 = 0.0
    for j in countup(0'i64,self.n):
        d += self.at(i,j) * vec.get(j.int64)
    