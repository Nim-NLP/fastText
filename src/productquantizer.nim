
import math
import random
import streams
# import ./float32
include system/ansi_c
include system/memory
import types
import strutils

proc distL2(x:ptr float32; y:ptr float32;  d:int32):float32 =
    for i in 0..<d:
        result += ((x[i][] - y[i][]) ^ 2)

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

proc assign_centroid*(self: ProductQuantizer; x: ptr float32; c0:ptr float32;code:ptr uint8;d: int32): float32 =
    var  c = c0
    var dis:float32 = distL2(x, c, d)
    code[] = 0
    var disij:float32
    var cp:int
    for j in 1..<ksub:
        c = c[d]
        disij = distL2(x,c, d)
        if (disij < dis):
            code[] = (uint8)j
            dis = disij
    return dis

proc Estep*(self: ProductQuantizer; x: ptr float32;centroids:ptr float32; codes: ptr uint8; d: int32;n: int32) =
    for i in 0..<n:
        discard self.assign_centroid(x[i * d] ,centroids,codes[i], d)

proc MStep*(self: ProductQuantizer; x0: ptr float32;centroids:ptr float32;codes:ptr uint8; d: int32; n: int32) =
    var nelts = newSeq[int32](ksub)
    
    nimSetMem(centroids, 0, sizeof(float32) * d * ksub)
    var x:ptr float32 = x0[0]
    var k:ptr uint8
    var c:ptr float32
    for i in 0..<n:
        k = codes[i]
        c = centroids[k[].int32 * d]
        for j in 0..<d:
            c[j][] += x[j][]
        nelts[k[]] += 1
        x[] += d.float32
    var c2 = centroids
    var z:int32
    for k in 0..<ksub:
        z = nelts[k]
        if (z != 0) :
            for j in 0..<d:
                c2[j][] = (c2[j][].int32 / z)
        c2[] += d.float32

    var m:int32
    var sign:int32
    var rng1 = self.rng
    for k in 0..<ksub:
        if (nelts[k] == 0):
            m = 0
            while (rng1.rand(1.0) * (n - ksub).toFloat >= cast[float](nelts[m] - 1)) :
                m = (m + 1) mod ksub
            nimCopyMem(centroids[k.int32 * d],centroids[m*d],sizeof(float32)*d)
            for j in 0'i32..<d:
                sign = (j mod 2) * 2 - 1;
                centroids[k.int32 * d + j][] += (sign.float32 * eps)
                centroids[m * d + j][] -= (sign.float32 * eps)
            
            nelts[k] = nelts[m] div 2
            nelts[m] -= nelts[k]

proc kmeans(self:var ProductQuantizer;x:ptr float32;c:ptr float32;n:int32;d:int32) =
    var perm = newSeq[int32](n)
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
        self.Estep(x,c,codes[0].addr,d,n)
        self.MStep(x,c,codes[0].addr,d,n)


proc train*(self:var ProductQuantizer;n:int32;norms:ptr float32) =
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
        self.kmeans(xslice.idata[0].addr, self.get_centroids(m.int32, 0),np,d)

proc compute_code*(self:var ProductQuantizer; x:ptr float32; code: ptr uint8) {.noSideEffect.} =
    var d = self.dsub
    var i:int32
    for m in 0..<self.nsubq:
        if m == self.nsubq - 1:
            d = self.lastdsub
        discard self.assign_centroid(x[m * self.dsub],self.get_centroids(m.int32, 0),code[m], d)

proc compute_codes*(self:var ProductQuantizer; x:ptr float32;codes:ptr uint8; n: int32) {.noSideEffect.} =
    for i in 0..<n:
        self.compute_code(x[i*self.dim],codes[i*self.nsubq])
    
# proc save*(this: var ProductQuantizer; a2: var ostream) {.stdcall, importcpp: "save",
#     header: headerproductquantizer.}

proc load*(self: var ProductQuantizer; a2: var Stream) =
    discard a2.readData(self.dim.addr,sizeof(self.dim))
    discard a2.readData(self.nsubq.addr,sizeof(self.nsubq))
    discard a2.readData(self.dsub.addr,sizeof(self.dsub))
    discard a2.readData(self.lastdsub.addr,sizeof(self.lastdsub))
    self.centroids.setLen(self.dim * ksub)
    for i in 0..<self.centroids.len():
        discard a2.readData(self.centroids[i].addr,sizeof(float32))