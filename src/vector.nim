import math
import ./types
import ./matrix
import ./qmatrix

proc zero*(self: var Vector) =
    for i in 0..<self.size():
        self.idata[i] = 0.0

proc mul*(self: var Vector; a: float32) =
    for i in 0..<self.size():
        self.idata[i] *= a

proc norm*(self: Vector): float32 =
    var sum: float32 = 0
    for i in 0..<self.size():
        sum += self.idata[i] * self.idata[i]
    result = sqrt(sum)

proc addVector*(self: var Vector; source: Vector) =
    assert(self.size() == source.size())
    for i in 0..<self.size():
        self.idata[i] += source.idata[i]

proc addVector*(self: var Vector; source: Vector; s: float32) =
    assert(self.size() == source.size())
    for i in 0..<self.size():
        self.idata[i] += s * source.idata[i]

proc addRow*(self: var Vector; A: Matrix; i: int64) =
    assert(i >= 0);
    assert(i < A.size(0'i64));
    assert(self.size() == A.size(1'i64))
    for j in 0..<A.size(1):
        self.idata[j] += A.at(i, j)

proc addRow*(self: var Vector; A: Matrix; i: int64; a: float32) =
    assert(i >= 0);
    assert(i < A.size(0'i64));
    assert(self.size() == A.size(1'i64));
    for j in 0..<A.size(1):
        self.idata[j] += a * A.at(i, j)

proc addRow*(self: var Vector; A: var QMatrix; i: int64; a: float32) =
    assert(i >= 0)
    A.addToVector(self, i.int32);

proc addRow*(self: var Vector; A: var QMatrix; i: int64; ) =
    assert(i >= 0);
    A.addToVector(self, i.int32);

proc mul*(self: var Vector; A: Matrix; vec: var Vector) =
    assert(A.size(0) == self.size());
    assert(A.size(1) == vec.size());
    for i in 0..<self.size():
        self.idata[i] = A.dotRow(vec, i.int32)

proc mul*(self: var Vector; A: var QMatrix; vec: var Vector) =
    assert(A.getM() == self.size())
    assert(A.getN() == vec.size())
    for i in 0..<self.size():
        self.idata[i] = A.dotRow(vec, i)

proc argmax*(self: var Vector): int64 =
    var
        max: float32 = self.idata[0]
    var i = 1
    while i < self.size():
        if self.idata[i] > max:
            max = self.idata[i];
            result = i
        inc i
