
import math
import random
import streams
# import ./float32
include system/ansi_c
include system/memory
import types



proc `[]`*(self:ptr float32,key:int):uint8 = 
  let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self)
  (uint8)a[key]

proc distL2(x: var Vector;xpost:int; y:ptr float32;  d:int32):float32 =
    var dist:float32  = 0.float32
    var i = 0
    while i < d:
        dist += ((x[i][] - y[i].float32).int ^ 2).float32
    return dist

# proc constructProductQuantizer*(): ProductQuantizer {.stdcall, constructor,
#     importcpp: "fasttext::ProductQuantizer(@)", header: headerproductquantizer.}
proc initProductQuantizer*(dim: int32; dsub: int32): ProductQuantizer =
    result.dim = dim
    result.dsub = dsub
    result.nsubq = result.dim div result.dsub
    result.centroids.setLen(dim * ksub)
    # rng(seed_)
    result.lastdsub = dim mod dsub
    if (result.lastdsub == 0):
        result.lastdsub = dsub
    else:
        inc result.nsubq

# proc `[]=`(self:ptr uint8,key:int,val:Natural){.discardable.} = 
#     let a:ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self)
#     a[key] = (uint8)val

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

proc train*(self:  ProductQuantizer; n: cint; x: ptr float32) =
    if n < ksub:
        raise newException(ValueError,"Matrix too small for quantization, must have at least " & $ksub & " rows")

proc compute_code*(self: ProductQuantizer; x: var Vector;xpos:int; code: var seq[uint8],codePos:int) {.noSideEffect.} =
    var d = self.dsub
    var i:int32
    for m in 0..<self.nsubq:
        if m == self.nsubq - 1:
            d = self.lastdsub
        i = self.getCentroidsPosition(m.int32,0'u8)
        discard self.assign_centroid(x,m * self.dsub,self.centroids[i],code, m.int32 , d)

proc compute_codes*(self: ProductQuantizer; x: var Vector;xpos:int; code: var seq[uint8],codePos:int; n: int32) {.noSideEffect.} =
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