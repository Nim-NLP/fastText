
import math
import random
import streams
# import ./float32
include system/ansi_c
include system/memory
import types





# proc distL2(x:ptr float32, y:ptr float32,  d:int32):float32 =
#     var dist:float32  = 0.float32
#     var i = 0
#     while i < d:
#         dist += ((x[i] - y[i]).int ^ 2).float32
#     return dist

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
# proc get_centroids*(self: ProductQuantizer; m: int32; i: uint8): ptr float32 {.
#     noSideEffect, stdcall, importcpp: "get_centroids",
#     header: headerproductquantizer.}
# proc assign_centroid*(self: ProductQuantizer; x:float32; c0:  float32; code: ptr uint8;d: int32): float32 =
#     var  c:float32 = c0
#     var dis:float32 = distL2(x.unsafeAddr, c.addr, d);
#     code[0] = 0;
#     var j = 1
#     var disij:float32
#     while j < ksub:
#         c += cast[float32](d)
#         disij = distL2(x.unsafeAddr, c.addr, d);
#         if (disij < dis) :
#             code[0] = (uint8)j;
#             dis = disij;
#         inc j
    
#     return dis;
                     

# proc Estep*(self: ProductQuantizer; x: float32; centroids: ptr float32; codes: ptr uint8; d: int32;n: int32) =
#     var i = 0
#     var code:uint8
#     while i < n:
#         code =  codes[] + i.uint8
#         discard self.assign_centroid(x + cast[float32](i * d), centroids[],code.addr, d);
#         inc i

# proc train*(this: var ProductQuantizer; a2: cint; a3: ptr float32) {.stdcall,
#     importcpp: "train", header: headerproductquantizer.}

# proc compute_code*(this: ProductQuantizer; a2: ptr float32; a3: ptr uint8) {.noSideEffect,
#     stdcall, importcpp: "compute_code", header: headerproductquantizer.}
# proc compute_codes*(this: ProductQuantizer; a2: ptr float32; a3: ptr uint8; a4: int32) {.
#     noSideEffect, stdcall, importcpp: "compute_codes",
#     header: headerproductquantizer.}
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