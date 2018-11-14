import math
# import ./productquantizer

import ./types
import ./matrix
import ./qmatrix

proc zero*(self: var Vector) =
    for i in countup(0,self.idata.len):
        self.idata[i] = 0.0

{.this: self.}
proc mul*(self: var Vector; a: float32) =
    for i in  0..<self.size():
        idata[i] *= a

{.this: self.}
proc norm*(self: Vector): float32 =
    var sum:float32 = 0
    for i in 0..<self.size():
        sum += idata[i] * idata[i]
    result = sqrt(sum)

{.this: self.}
proc addVector*(self: var Vector; source: Vector) =
    assert(self.size() == source.size())
    for i in 0..<self.size():
        idata[i] += source.idata[i]

{.this: self.}
proc addVector*(self: var Vector; source: Vector; s: float32) =
    assert(self.size() == source.size())
    for i in 0..<self.size():
        idata[i] += s * source.idata[i]

{.this: self.}
proc addRow*(self: var Vector; A: Matrix; i: int64) =
    assert(i >= 0);
    assert(i < A.size(0'i64));
    assert(size() == A.size(1'i64))
    for j in  0..<A.size(1):
        idata[j] += A.at(i, j)

{.this: self.}
proc addRow*(self: var Vector; A: Matrix; i: int64;a:float32) =
    assert(i >= 0);
    assert(i < A.size(0'i64));
    assert(size() == A.size(1'i64));
    var j = 0
    while j < A.size(1):
        idata[j] += a * A.at(i, j)
        inc j

proc addRow*(self: var Vector; A: QMatrix; i: int64; a: float32) =
    assert(i >= 0)
    A.addToVector(self, i.int32);

proc addRow*(self: var Vector; A:var QMatrix; i: int64;) =
    assert(i >= 0);
    A.addToVector(self, i.int32);

proc mul*(self: var Vector; A: Matrix; vec:var Vector) =
    assert(A.size(0) == self.size());
    assert(A.size(1) == vec.size());
    for  i in 0..<size():
        idata[i] = A.dotRow(vec, i.int64)

proc mul*(self: var Vector; A: QMatrix; vec:var Vector) =
    assert(A.getM() == self.size())
    assert(A.getN() == vec.size())
    for i in 0..<size():
        idata[i] = A.dotRow(vec, i)

proc argmax*(self: var Vector): int64 =
    var 
        max:float32 = self.idata[0]
    var i = 1
    while i < size():
        if self.idata[i] > max:
            max = self.idata[i];
            result = i
        inc i