import math
import ./productquantizer

import ./types
import ./matrix
import ./qmatrix

proc constructVector*(a1: int64): Vector =
    result.idata = newSeq[float32](a1)

proc constructVector*(a1: Vector): Vector =
    result = a1

proc zero*(self: var Vector) =
    for i in countup(0,self.idata.len):
        self.data[i] = 0.0

{.this: self.}
proc mul*(self: var Vector; a: float32) =
    var i = 0
    while i < self.size():
        idata[i] *= a;
        inc i

{.this: self.}
proc norm*(self: Vector): float32 =
    var sum:float32 = 0;
    var i = 0
    while i < self.size():
        sum += idata[i] * idata[i]
        inc i
    sqrt(sum)

{.this: self.}
proc addVector*(self: var Vector; source: Vector) =
    doAssert(self.size() == source.size())
    var i = 0
    while i < self.size():
        idata[i] += source.idata[i]
        inc i

{.this: self.}
proc addVector*(self: var Vector; source: Vector; s: float32) =
    doAssert(self.size() == source.size())
    var i = 0
    while i < self.size():
        idata[i] += s * source.idata[i]
        inc i

{.this: self.}
proc addRow*(self: var Vector; A: Matrix; i: int64) =
    doAssert(i >= 0);
    doAssert(i < A.size(0'i64));
    doAssert(size() == A.size(1'i64));
    var j = 0
    while j < A.size(1):
        idata[j] += A.at(i, j);
        inc j

{.this: self.}
proc addRow*(self: var Vector; A: Matrix; i: int64;a:float32) =
    doAssert(i >= 0);
    doAssert(i < A.size(0'i64));
    doAssert(size() == A.size(1'i64));
    var j = 0
    while j < A.size(1):
        idata[j] += a * A.at(i, j)
        inc j

proc addRow*(self: var Vector; A: QMatrix; i: int64; a: float32) =
    doAssert(i >= 0)
    A.addToVector(self, i.int32);

proc addRow*(self: var Vector; A:var QMatrix; i: int64;) =
    doAssert(i >= 0);
    A.addToVector(self, i.int32);

proc mul*(self: var Vector; A: Matrix; vec: Vector) =
    doAssert(A.size(0) == self.size());
    doAssert(A.size(1) == vec.size());
    var i = 0
    while i < size():
        idata[i] = A.dotRow(vec, i);
        inc i

proc mul*(self: var Vector; A: QMatrix; vec: Vector) =
    doAssert(A.getM() == self.size());
    doAssert(A.getN() == vec.size());
    var i = 0
    while i < size():
        idata[i] = A.dotRow(vec, i);
        inc i

proc argmax*(self: var Vector): int64 =
    var 
        max:float32 = self.idata[0]
        argmax:int64 = 0;
    var i = 1
    while i < size():
        if self.idata[i] > max:
            max = self.idata[i];
            argmax = i;
        inc i