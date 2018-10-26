
import math
import random
# import ./float32
include system/memory

const nbits:int32 = 8;
const ksub:int32 = 1 shl nbits;
const max_points_per_cluster:int32 = 256;
const max_points:int32 = max_points_per_cluster * ksub
const seed:int32 = 1234;
const niter:int32 = 25;
const eps:float32 = 1e-7.float32;

type
  ProductQuantizer*  = object
    dim,nsubq,dsub,lastdsub:int32
    centroids:ref seq[ ptr float32]
    # std::minstd_rand rng;
  
proc `[]`*(self:ptr float32,key:int):uint8 = 
    let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self)
    (uint8)a[key]

proc distL2(x:ptr float32, y:ptr float32,  d:int32):float32 =
    var dist:float32  = 0.float32
    var i = 0
    while i < d:
        dist += ((x[i] - y[i]).int ^ 2).float32
    return dist;


# proc constructProductQuantizer*(): ProductQuantizer {.stdcall, constructor,
#     importcpp: "fasttext::ProductQuantizer(@)", header: headerproductquantizer.}
proc constructProductQuantizer*(dim: int32; dsub: int32): ProductQuantizer =
    result.dim = dim
    result.dsub = dsub
    result.nsubq = result.dim div result.dsub
    result.centroids[].setLen(dim * ksub)
    # rng(seed_)
    result.lastdsub = dim mod dsub
    if (result.lastdsub == 0):
        result.lastdsub = dsub
    else:
        inc result.nsubq

{.this: self.} 
proc get_centroids*(self: var ProductQuantizer; m: int32; i: uint8): ptr float32 =
    if (m == nsubq - 1) :
        return centroids[m * ksub * dsub + cast[int32](i) * lastdsub]
    return centroids[(m * ksub + cast[int32](i)) * dsub];

proc `[]`*(self:ptr uint8,key:int):uint8 = 
    let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self)
    (uint8)a[key]

proc `[]=`*(self:ptr uint8,key:int,val:Natural){.discardable.} = 
    let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self)
    a[key] = (uint8)val
# proc get_centroids*(self: ProductQuantizer; m: int32; i: uint8): ptr float32 {.
#     noSideEffect, stdcall, importcpp: "get_centroids",
#     header: headerproductquantizer.}
proc assign_centroid*(self: ProductQuantizer; x:float32; c0:  float32; code: ptr uint8;d: int32): float32 =
    var  c:float32 = c0
    var dis:float32 = distL2(x.unsafeAddr, c.addr, d);
    code[0] = 0;
    var j = 1
    var disij:float32
    while j < ksub:
        c += cast[float32](d)
        disij = distL2(x.unsafeAddr, c.addr, d);
        if (disij < dis) :
            code[0] = (uint8)j;
            dis = disij;
        inc j
    
    return dis;
                     

proc Estep*(self: ProductQuantizer; x: float32; centroids: ptr float32; codes: ptr uint8; d: int32;n: int32) =
    var i = 0
    var code:uint8
    while i < n:
        code =  codes[] + i.uint8
        discard self.assign_centroid(x + cast[float32](i * d), centroids[],code.addr, d);
        inc i

proc `[]`*(self:var float32,key:int): ptr uint8 = 
    let a:ptr UncheckedArray[ uint8] = cast[ptr UncheckedArray[ uint8]](self.addr)
    a[key].addr

proc exchange*(self:ptr float32,dest,src:int) =
    let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self)
    a[dest] = a[src]
    
{.this: self.} 
proc MStep*(self: var ProductQuantizer; x0: ptr float32; centroids: ptr float32; codes: ptr uint8; d: int32; n: int32) =
    var nelts = newSeq[int32](ksub)
    nimSetMem(centroids, 0, sizeof(centroids) * d * ksub)
    var x:float32 = x0[];
    var  
        k:int
        c: float32
        z:ptr UncheckedArray[byte]
    for i in 0..<n:
        z = cast[ptr UncheckedArray[byte]](codes)
        k = z[i].int;
        c = (centroids[] + (k * d).float32)
        for j in 0..<d:
            c[j][] += x[j][]
        nelts[k] += 1;
        x += d.float32
    c = centroids[]
    var z1:float32
    for k in 0..<ksub:
        z1 = nelts[k].float32
        if (z1 != 0) :
            for j in 0..<d:
                c[j][]  /=  z1
        c += d.float32
    k = 0
    randomize()
    var m:int32
    var j2:int32
    var sign:int32
    while k < ksub:
        if (nelts[k] == 0):
            m = 0
            while (rand(1.0) * (n - ksub).toFloat >= cast[float](nelts[m] - 1)) :
                m = (m + 1) mod ksub
            # nimCopyMem(centroids[k * d].unsafeAddr)
            centroids.exchange(k * d,m * d)
            j2 = 0
            while j2 < d:
                sign = (j2 mod 2) * 2 - 1;
                centroids[k * d + j2] += sign * eps_;
                centroids[m * d + j2] -= sign * eps_;
                inc j2
            
            nelts[k] = nelts[m] / 2;
            nelts[m] -= nelts[k];
        
        inc k
        
proc kmeans*(this: var ProductQuantizer; x: ptr float32; c: ptr float32; n: int32; d: int32) =
    # std::vector<int32_t> perm(n,0);
    var perm = newSeq[int32](n)
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (auto i = 0; i < ksub_; i++) {
        memcpy (&c[i * d], x + perm[i] * d, d * sizeof(float32));
    }
    auto codes = std::vector<uint8_t>(n);
    for (auto i = 0; i < niter_; i++) {
        Estep(x, c, codes.data(), d, n);
        MStep(x, c, codes.data(), d, n);
    }
proc train*(this: var ProductQuantizer; a2: cint; a3: ptr float32) {.stdcall,
    importcpp: "train", header: headerproductquantizer.}
proc mulcode*(this: ProductQuantizer; a2: Vector; a3: ptr uint8; a4: int32; a5: float32): float32 {.
    noSideEffect, stdcall, importcpp: "mulcode", header: headerproductquantizer.}
proc addcode*(this: ProductQuantizer; a2: var Vector; a3: ptr uint8; a4: int32; a5: float32) {.
    noSideEffect, stdcall, importcpp: "addcode", header: headerproductquantizer.}
proc compute_code*(this: ProductQuantizer; a2: ptr float32; a3: ptr uint8) {.noSideEffect,
    stdcall, importcpp: "compute_code", header: headerproductquantizer.}
proc compute_codes*(this: ProductQuantizer; a2: ptr float32; a3: ptr uint8; a4: int32) {.
    noSideEffect, stdcall, importcpp: "compute_codes",
    header: headerproductquantizer.}
proc save*(this: var ProductQuantizer; a2: var ostream) {.stdcall, importcpp: "save",
    header: headerproductquantizer.}
proc load*(this: var ProductQuantizer; a2: var istream) {.stdcall, importcpp: "load",
    header: headerproductquantizer.}