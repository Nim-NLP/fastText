
import ./productquantizer
import ./types
import ./vector
import ./matrix
import streams
import strutils

proc quantize*(self: var QMatrix; matrix: Matrix)
proc quantizeNorm*(self:var QMatrix;norms:var Vector)

proc initQMatrix*(): QMatrix =
    result# = QMatrix(qnorm:false,m:0,n:0,codesize:0)

proc initQMatrix*(mat:var Matrix; dsub: int32; qnorm: bool): QMatrix =
    result.m = mat.size(0)
    result.n = mat.size(1)
    result.codesize = result.m.int32 * ((result.n.int32 + dsub - 1) div dsub)
    result.codes.setLen(result.codesize)
    result.pq = newProductQuantizer(result.n.int32,dsub)
    result.qnorm = qnorm
    if result.qnorm:
        result.norm_codes.setLen(result.m)
        result.npq = newProductQuantizer(1'i32,1'i32)
    result.quantize(mat)

proc quantizeNorm*(self:var QMatrix;norms:var Vector) =
    assert self.qnorm == true
    assert norms.size() == self.m
    self.npq[].train(self.m.int32, norms)
    self.npq[].compute_codes(norms,0'i32, self.norm_codes,0'i32, self.m.int32);

proc quantize*(self:var QMatrix;matrix:Matrix) =
    assert(self.m == matrix.size(0))
    assert(self.n == matrix.size(1))
    var temp = matrix
    var norms:Vector
    if self.qnorm:
        norms = initVector(temp.size(0))
        temp.l2NormRow(norms)
        temp.divideRow(norms)
        self.quantizeNorm(norms)

proc save*(self: var QMatrix; o: var Stream) =
    o.writeData(addr self.qnorm,sizeof(self.qnorm))
    o.writeData(addr self.m,sizeof(self.m))
    o.writeData(addr self.n,sizeof(self.n))
    o.writeData(addr self.codesize,sizeof(self.codesize))
    # o.writeData(addr codes.data(),codesize * sizeof(uint8))
    # pq.save(o)
    # if (qnorm) :
    #   o.write(norm_codes.data(), m * sizeof(uint8));
    #   npq.save(o);
    

proc load*(self: var QMatrix; a2: var Stream) =
    discard a2.readData(addr self.qnorm,sizeof(bool))
    discard a2.readData(addr self.m,sizeof(int64))
    discard a2.readData(addr self.n,sizeof(int64))
    discard a2.readData(addr self.codesize,sizeof(int32))
    self.codes.setLen(self.codesize)
    for j in 0..<self.codes.len :
        discard a2.readData(self.codes[j].addr, sizeof(uint8))
    self.pq = newProductQuantizer()
    self.pq[].load(a2)
    debugEcho "self.qnorm",self.qnorm
    if self.qnorm:
        self.norm_codes.setLen(self.m)
        for i in 0..<self.m:
            discard a2.readData(self.norm_codes[i].addr, sizeof(uint8))
        self.npq = newProductQuantizer()
        self.npq[].load(a2)