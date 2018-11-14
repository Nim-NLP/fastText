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
      pq*,npq*: ProductQuantizer

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

proc getM*(self: QMatrix): int64 =
    self.m

proc getN*(self: QMatrix): int64 =
    self.n

# proc quantizeNorm*(self: var QMatrix; norms: Vector) =
#     assert(self.qnorm == true)
#     assert(norms.size() == self.m )
#     let dataptr =  norms.data()
#     # npq.train(m_, dataptr)
#     # npq.compute_codes(dataptr, self.norm_codes.data(), m);

# proc quantize*(self: var QMatrix; matrix: Matrix) =
#     doassert(self.m == matrix.size(0));
#     doassert(self.n == matrix.size(1));
#     let temp  = matrix
#     # if (self.qnorm) :
#     #     Vector norms(temp.size(0));
#     #     temp.l2NormRow(norms);
#     #     temp.divideRow(norms);
#     #     quantizeNorm(norms);

#     # auto dataptr = temp.data();
#     # pq_->train(m_, dataptr);
#     # pq_->compute_codes(dataptr, codes_.data(), m_);
proc addToVector*(self: QMatrix; x: var Vector; t: int32) =
    var norm:float32 = 1
    # if self.qnorm:
    #     norm = npq.get_centroids(0, norm_codes_[t])[0]
    # pq.addcode(x, codes_.data(), t, norm);
proc dotRow*(self: QMatrix; vec: Vector; i: int64): float32 =
    doassert(i >= 0);
    doassert(i < self.m)
    doassert(vec.size() == self.n)
    var norm:float32 = 1
    # if (qnorm_) {
    #     norm = npq_->get_centroids(0, norm_codes_[i])[0];
    # }
    # return pq_->mulcode(vec, codes_.data(), i, norm);

        
    