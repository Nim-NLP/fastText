
import ./productquantizer
import ./types
import ./vector
import ./matrix
import streams

# proc quantize*(self: var QMatrix; matrix: Matrix)

proc initQMatrix*(): QMatrix =
    result = QMatrix(qnorm:false,m:0,n:0,codesize:0)

proc initQMatrix*(mat:var Matrix; dsub: int32; qnorm: bool): QMatrix =
    let 
        m = mat.size(0)
        n = mat.size(1)
    let codesize = m.int32 * ((n.int32 + dsub - 1) div dsub)
    result = QMatrix(qnorm:false,m:m.int64,n:n,codesize:codesize.int32)
    result.codes.setLen(codesize)
    result.pq = initProductQuantizer(n.int32,dsub)
    if result.qnorm:
        result.norm_codes.setLen(m)
        result.npq = initProductQuantizer(1'i32,1'i32)
    # result.quantize(mat);

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
    discard a2.readData(addr self.qnorm,sizeof(self.qnorm))
    discard a2.readData(addr self.m,sizeof(self.m))
    discard a2.readData(addr self.n,sizeof(self.n))
    discard a2.readData(addr self.codesize,sizeof(self.codesize))
    self.codes = newSeq[uint8](self.codesize)
    for j in 0..<self.codes.len :
        discard a2.readData(self.codes[j].addr, sizeof(uint8))
    self.pq = ProductQuantizer()
    self.pq.load(a2)
    if self.qnorm:
        for i in 0..<self.norm_codes.len:
            discard a2.readData(self.norm_codes[i].addr, sizeof(uint8))
        self.npq = ProductQuantizer()
        self.npq.load(a2)