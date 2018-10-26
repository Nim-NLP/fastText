import ./productquantizer
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
    doassert(dim == 0 or dim == 1 )
    result = if dim == 0 : self.m else : self.n

type
    Vector*  = object
        idata*:seq[ float32]


proc size*(self: Vector): int64 =
    self.idata.len

proc data*(self: var Vector): ptr seq[float32] =
    addr self.idata

proc data*(self: Vector): ptr seq[float32] =
    self.idata.unSafeAddr

proc `[]`*(self: var Vector; i: int64): ptr float32 =
    let a:ptr UncheckedArray[float32] = cast[ptr UncheckedArray[float32]](self.data)
    result = a[i.int32].addr

proc get*(self: Vector; i: int64): float32 =
    self.idata[i.int32]