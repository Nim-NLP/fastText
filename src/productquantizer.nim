
import math
import random
import streams
# import ./float32
include system/ansi_c
include system/memory
import types
import strutils

proc distL2(x: var Vector;xpos:int; y:ptr float32;  d:int32):float32 =
    var xv = x[xpos]
    for i in 0..<d:
        result += ((xv[i][] - y[i][]).int ^ 2).float32

proc initProductQuantizer*(): ProductQuantizer =
    result.rng = initRand(seed)
    
    
proc newProductQuantizer*():ref ProductQuantizer =
    result = new ProductQuantizer
    result.rng = initRand(seed)

proc initProductQuantizer*(dim: int32; dsub: int32): ProductQuantizer =
    result.dim = dim
    result.nsubq = result.dim div result.dsub
    result.dsub = dsub
    result.centroids.setLen(dim * ksub)
   
    result.rng = initRand(seed)
    result.lastdsub = dim mod dsub
    if (result.lastdsub == 0):
        result.lastdsub = dsub
    else:
        inc result.nsubq

proc newProductQuantizer*(dim: int32; dsub: int32):ref ProductQuantizer =
    result = new ProductQuantizer
    result.dim = dim
    result.nsubq = result.dim div result.dsub
    result.dsub = dsub
    result.centroids.setLen(dim * ksub)
    result.rng = initRand(seed)
   
    result.lastdsub = dim mod dsub
    if (result.lastdsub == 0):
        result.lastdsub = dsub
    else:
        inc result.nsubq

proc assign_centroid*(self: ProductQuantizer; x: var Vector;xpos:int; c0: float32; codes: var seq[uint8],codePos:int;d: int32): float32 =
    var  c:float32 = c0
    var code = codes[codePos].addr
    var dis:float32 = distL2(x,xpos, c.addr, d)
    code[] = 0

    var disij:float32
    for j in 1..<ksub:
        c += d.float32
        disij = distL2(x,xpos, c.addr, d)
        if (disij < dis):
            code[] = (uint8)j
            dis = disij
    return dis

proc Estep*(self: ProductQuantizer; x: var Vector;xpos:int; centroidPos:int32; codes: var seq[uint8];codePos:int32; d: int32;n: int32) =
    for i in 0..<n:
        discard self.assign_centroid(x ,xpos + i * d,self.centroids[centroidPos],codes,i, d)

proc Estep*(self: ProductQuantizer; x: var Vector;xpos:int;centroids:ptr float32; codes: var seq[uint8];codePos:int32; d: int32;n: int32) =
    for i in 0..<n:
        discard self.assign_centroid(x ,xpos + i * d,centroids[],codes,i, d)

proc MStep*(self: ProductQuantizer; x0: var Vector;xpos:int;centroidPos:int; codesSeq: var seq[uint8];codePos:int32; d: int32; n: int32) =
    var nelts = newSeq[int32](ksub)
    var codes = codesSeq[codePos].addr
    nimSetMem(self.centroids[centroidPos].unsafeAddr, 0, sizeof(float32) * d * ksub)
    var x:ptr float32 = x0[0]
    var k:uint8
    var c:ptr float32
    for i in 0..<n:
        k = codesSeq[i]
        c = self.centroids[centroidPos + k.int32 * d].unsafeAddr
        for j in 0..<d:
            c[j][] += x[j][]
        nelts[k] += 1
        x[] += d.float32
    var c2 = self.centroids[centroidPos].unsafeAddr
    var z:int32
    for k in 0..<ksub:
        z = nelts[k]
        if (z != 0) :
            for j in 0..<d:
                c2[j][] = (c2[j][].int32 / z).uint8
        c2[] += d.float32

    var m:int32
    var sign:int32
    var rng1 = self.rng
    for k in 0..<ksub:
        if (nelts[k] == 0):
            m = 0
            while (rng1.rand(1.0) * (n - ksub).toFloat >= cast[float](nelts[m] - 1)) :
                m = (m + 1) mod ksub
            nimCopyMem(self.centroids[centroidPos+k.int32 * d].unsafeAddr,self.centroids[centroidPos+m*d].unsafeAddr,sizeof(float32)*d)
            for j in 0'i32..<d:
                sign = (j mod 2) * 2 - 1;
                self.centroids[centroidPos+k.int32 * d + j].unsafeAddr[] += (sign.float32 * eps)
                self.centroids[centroidPos+m * d + j].unsafeAddr[] -= (sign.float32 * eps)
            
            nelts[k] = nelts[m] div 2
            nelts[m] -= nelts[k]

proc kmeans(self:ProductQuantizer;x:var Vector;centeroidPos:int;n:int32;d:int32) =
    var perm = newSeq[int32](n)
    var c = self.centroids[centeroidPos].unsafeAddr
    var i = 0'i32
    while i < n:
        perm[i] = i
        inc i
    var r = self.rng
    r.shuffle(perm)
    for i in 0..<ksub:
        nimCopyMem(self.centroids[i.int32 * d].unsafeAddr,perm[i*d].unsafeAddr,sizeof(float32)*d)
    var codes = newSeq[uint8](n)
    for i in 0..<niter:
        self.Estep(x,0,c,codes,0,d,n)
        self.MStep(x,0,centeroidPos,codes,0,d,n)


proc train*(self:var ProductQuantizer;n:int32;norms:var Vector) =
    debugEcho "train"
    if n < ksub:
        raise newException(ValueError,"Matrix too small for quantization, must have at least " & $ksub & " rows")
    var perm = newSeq[int32](n)
    for i in 0'i32..<n:
        perm[i] = i
    var d = self.dsub
    var np = min(n,max_points)
    var xslice = initVector(np * self.dsub)
    var i:int32
    for m in 0..<self.nsubq:
        if m == self.nsubq - 1:
            d = self.lastdsub
        if np != n :
            self.rng.shuffle(perm)
        for j in 0..<np:
            xslice.idata[j*d] = norms[j*self.dim + m * self.dsub][]
        i = self.getCentroidsPosition(m.int32,0'u8)
        self.kmeans(xslice,i,np,d)

proc compute_code*(self: ProductQuantizer; x: var Vector;xpos:int; code: var seq[uint8],codePos:int) {.noSideEffect.} =
    var d = self.dsub
    var i:int32
    for m in 0..<self.nsubq:
        if m == self.nsubq - 1:
            d = self.lastdsub
        i = self.getCentroidsPosition(m.int32,0'u8)
        discard self.assign_centroid(x,xpos+m * self.dsub,self.centroids[i],code, codePos+m.int32 , d)

proc compute_codes*(self: ProductQuantizer; x: var Vector;xpos:int32; code: var seq[uint8],codePos:int32; n: int32) {.noSideEffect.} =
    for i in 0..<n:
        self.compute_code(x,xpos + i*self.dim,code,codePos+i*self.nsubq)
    
# proc save*(this: var ProductQuantizer; a2: var ostream) {.stdcall, importcpp: "save",
#     header: headerproductquantizer.}

proc load*(self: var ProductQuantizer; a2: var Stream) =
    

    discard a2.readData(self.dim.addr,sizeof(self.dim))
    discard a2.readData(self.nsubq.addr,sizeof(self.nsubq))
    discard a2.readData(self.dsub.addr,sizeof(self.dsub))
    discard a2.readData(self.lastdsub.addr,sizeof(self.lastdsub))
    debugEcho self.dim
    debugEcho self.dim * ksub
    self.centroids.setLen(self.dim * ksub)
    for i in 0..<self.centroids.len():
        discard a2.readData(self.centroids[i].addr,sizeof(float32))