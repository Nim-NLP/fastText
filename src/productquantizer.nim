
import math
import random
import streams
# import ./float32
include system/ansi_c
include system/memory
import types
import strutils


proc `[]`*(self:ptr float32,key:int):ptr uint8 = 
  let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self)
  a[key].unsafeaddr

proc distL2(x: var Vector;xpost:int; y:ptr float32;  d:int32):float32 =
    var dist:float32  = 0.float32
    var i = 0
    while i < d:
        dist += ((x[i][] - y[i][].float32).int ^ 2).float32
    return dist

proc initProductQuantizer*(): ProductQuantizer =
    result.seed = 1234
    result.nbits = 8
    result.ksub = 1'i32 shl result.nbits
    result.max_points_per_cluster = 256
    result.max_points = result.max_points_per_cluster * result.ksub
    result.niter = 25
    result.eps = 1e-7
    result.rng = initRand(result.seed)


proc initProductQuantizer*(dim: int32; dsub: int32): ProductQuantizer =
    result.dim = dim
    result.dsub = dsub
    result.nsubq = result.dim div result.dsub
    result.centroids.setLen(dim * ksub)
    result.seed = 1234
    result.nbits = 8
    result.ksub = 1'i32 shl result.nbits
    result.max_points_per_cluster = 256
    result.max_points = result.max_points_per_cluster * result.ksub
    result.niter = 25
    result.eps = 1e-7
    result.rng = initRand(result.seed)
    result.lastdsub = dim mod dsub
    if (result.lastdsub == 0):
        result.lastdsub = dsub
    else:
        inc result.nsubq

proc assign_centroid*(self: ProductQuantizer; x: var Vector;xpos:int; c0: float32; code: var seq[uint8],codePos:int;d: int32): float32 =
    var  c:float32 = c0
    var dis:float32 = distL2(x,xpos, c.addr, d);
    code[0] = 0;
    var j = 1
    var disij:float32
    while j < ksub:
        c += cast[float32](d)
        disij = distL2(x,xpos, c.addr, d);
        if (disij < dis) :
            code[0] = (uint8)j;
            dis = disij;
        inc j
    
    return dis;

proc Estep*(self: ProductQuantizer; x: var Vector;xpos:int; centroids:var seq[float32];centroidPos:int32; codes: var seq[uint8];codePos:int32; d: int32;n: int32) =
    var i = 0
    while i < n:
        discard self.assign_centroid(x ,xpos,self.centroids[centroidPos],codes,i, d);
        inc i

proc Estep*(self: ProductQuantizer; x: var Vector;xpos:int;centroids:float32; codes: var seq[uint8];codePos:int32; d: int32;n: int32) =
    var i = 0
    while i < n:
        discard self.assign_centroid(x ,xpos,centroids,codes,i, d);
        inc i

{.this: self.} 
proc MStep*(self: ProductQuantizer; x0: var Vector;xpos:int;centroids:ptr float32; codes: var seq[uint8];codePos:int32; d: int32; n: int32) =
    var nelts = newSeq[int32](ksub)
    nimSetMem(centroids, 0, sizeof(float32) * d * ksub)
    var x:ptr float32 = x0[0]
    var k:uint8
    var c:float32
    var z:ptr UncheckedArray[byte]
    for i in 0..<n:
        z = cast[ ptr UncheckedArray[byte] ](c)
        k = codes[i]
        c = centroids[] + k.float32 * d.float32
        for j in 0..<d:
            z[j] += x[j][]
        nelts[k] += 1
        x[] += d.float32
    var c2 = centroids
    var z1:int32
    for k in 0..<ksub:
        z1 = nelts[k]
        if (z1 != 0) :
            for j in 0..<d:
                z = cast[ ptr UncheckedArray[byte] ](c)
                z[j] = (c2[j][].int32 / z1).uint8
        c += d.float32

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
                centroids[k.int32 * d + j][] += (sign.float32 * self.eps).uint8
                centroids[m * d + j][] -= (sign.float32 * self.eps).uint8
            
            nelts[k] = nelts[m] div 2
            nelts[m] -= nelts[k]

proc kmeans(self:ProductQuantizer;x:var Vector;c:float32;n:int32;d:int32) =
    var perm = newSeq[int32](n)
    var i = 0'i32
    while i < n:
        perm[i] = i
        inc i
    var r = self.rng
    r.shuffle(perm)
    var codes = newSeq[uint8](n)
    for i in 0..<self.niter:
        self.Estep(x,0,c,codes,0,d,n)
        self.MStep(x,0,c.unsafeAddr,codes,0,d,n)


proc train*(self:var ProductQuantizer;n:int32;norms:var Vector) =
    if n < ksub:
        raise newException(ValueError,"Matrix too small for quantization, must have at least " & $ksub & " rows")
    var perm = newSeq[int32](n)
    var i = 0'i32
    while i < n:
        perm[i] = i
        inc i
    var d = self.dsub
    var np = min(n,self.max_points)
    var xslice = initVector(np * self.dsub)
    var z:int32
    for m in 0..<self.nsubq:
        if m == self.nsubq - 1:
            d = self.lastdsub
        if np != n :
            self.rng.shuffle(perm)
        for j in 0..<np:
            xslice.idata[j*d] = norms[j*self.dim + m * self.dsub][]
        z = self.getCentroidsPosition(m.int32,0'u8)
        self.kmeans(xslice,self.centroids[i],np,d)

proc compute_code*(self: ProductQuantizer; x: var Vector;xpos:int; code: var seq[uint8],codePos:int) {.noSideEffect.} =
    var d = self.dsub
    var i:int32
    for m in 0..<self.nsubq:
        if m == self.nsubq - 1:
            d = self.lastdsub
        i = self.getCentroidsPosition(m.int32,0'u8)
        discard self.assign_centroid(x,m * self.dsub,self.centroids[i],code, m.int32 , d)

proc compute_codes*(self: ProductQuantizer; x: var Vector;xpos:int32; code: var seq[uint8],codePos:int32; n: int32) {.noSideEffect.} =
    for i in 0..<n:
        self.compute_code(x,i*self.dim,code,i*self.nsubq)
    
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