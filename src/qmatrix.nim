
import ./productquantizer
import ./types
import ./vector
import ./matrix
import streams

proc initQMatrix*(): QMatrix =
    result = QMatrix(qnorm:false,m:0,n:0,codesize:0)

proc initQMatrix*(mat:var Matrix; dsub: int32; qnorm: bool): QMatrix =
    let 
        m = mat.size(0)
        n = mat.size(1)
    let codesize = m.int32 * ((n.int32 + dsub - 1) div dsub)
    result = QMatrix(qnorm:false,m:m.int64,n:n,codesize:codesize.int32)
    result.codes.setLen(codesize)
    # pq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer(n_, dsub));
    # if (qnorm_) {
    #     norm_codes_.resize(m_);
    #     npq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer(1, 1));
    # }
    # quantize(mat);

proc getM*(self: QMatrix): int64 =
    self.m

proc getN*(self: QMatrix): int64 =
    self.n

proc quantizeNorm*(self: var QMatrix; norms: Vector) =
    # assert(qnorm_);
    doAssert(norms.size() == self.m )
    let dataptr =  norms.data()
    # npq.train(m_, dataptr);
    # npq.compute_codes(dataptr, self.norm_codes.data(), m);

proc quantize*(self: var QMatrix; matrix: Matrix) =
    doassert(self.m == matrix.size(0));
    doassert(self.n == matrix.size(1));
    let temp  = matrix
    # if (self.qnorm) :
    #     Vector norms(temp.size(0));
    #     temp.l2NormRow(norms);
    #     temp.divideRow(norms);
    #     quantizeNorm(norms);

    # auto dataptr = temp.data();
    # pq_->train(m_, dataptr);
    # pq_->compute_codes(dataptr, codes_.data(), m_);
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

{.this: self.} 
proc save*(self: var QMatrix; o: var Stream) =
    o.writeData(addr qnorm,sizeof(qnorm))
    o.writeData(addr m,sizeof(m))
    o.writeData(addr n,sizeof(n))
    o.writeData(addr codesize,sizeof(codesize))
    # o.writeData(addr codes.data(),codesize * sizeof(uint8))
    # pq.save(o)
    # if (qnorm) :
    #   o.write(norm_codes.data(), m * sizeof(uint8));
    #   npq.save(o);
    
{.this: self.} 
proc load*(self: var QMatrix; a2: var Stream) =
    discard a2.readData(addr qnorm,sizeof(qnorm))
    discard a2.readData(addr m,sizeof(m))
    discard a2.readData(addr n,sizeof(n))
    discard a2.readData(addr codesize,sizeof(codesize))
    discard a2.readData(addr m,sizeof(m))
    discard a2.readData(addr m,sizeof(m))
    codes = newSeq[uint8](codesize);
    # in.read((char*) codes_.data(), codesize_ * sizeof(uint8_t));
    # pq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer());
    # pq_->load(in);
    # if (qnorm_) {
    #   norm_codes_ = std::vector<uint8_t>(m_);
    #   in.read((char*) norm_codes_.data(), m_ * sizeof(uint8_t));
    #   npq_ = std::unique_ptr<ProductQuantizer>( new ProductQuantizer());
    #   npq_->load(in);
    # }