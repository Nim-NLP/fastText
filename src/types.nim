import math
import random, tables, streams, strutils

const nbits*: int32 = 8;
const ksub*: int32 = 1 shl nbits;
const max_points_per_cluster*: int32 = 256;
const max_points*: int32 = max_points_per_cluster * ksub
const seed*: int32 = 1234;
const niter*: int32 = 25;
const eps*: float32 = 1e-7.float32;

type
  model_name* = enum
    cbow = 1, sg, sup


type
  loss_name* = enum
    hs = 1, ns, softmax


type
  Args* = object
    input*: string
    output*: string
    lr*: float64
    lrUpdateRate*: cint
    dim*: cint
    ws*: cint
    epoch*: cint
    minCount*: cint
    minCountLabel*: cint
    neg*: cint
    wordNgrams*: cint
    loss*: loss_name
    model*: model_name
    bucket*: cint
    minn*: cint
    maxn*: cint
    thread*: cint
    t*: float64
    label*: string
    verbose*: cint
    pretrainedVectors*: string
    saveOutput*: bool
    qout*: bool
    retrain*: bool
    qnorm*: bool
    cutoff*: csize
    dsub*: csize



type
  id_type* = int32
  entry_type* = enum # enum class entry_type : int8_t {word=0, label=1};
    word = 0, label = 1
  entry* = object
    word*: string
    count*: int64
    entry_type*: entry_type
    subwords*: seq[int32]

  Dictionary* = object
    args*: ref Args
    word2int*: seq[int32]
    words*: seq[entry]
    pdiscard*: seq[float32]
    size*: int32
    nwords*: int32
    nlabels*: int32
    ntokens*: int64
    pruneidxsize*: int64
    pruneidx*: Table[int32, int32]




type
  ProductQuantizer* = object
    dim*, nsubq*, dsub*, lastdsub*: int32
    centroids*: seq[float32]
    rng*: Rand

type
  Matrix* = object
    idata*: seq[float32]
    m*, n*: int64
  QMatrix* = object
    qnorm*: bool
    m*, n*: int64
    codesize*: int32
    codes*: seq[uint8]
    norm_codes*: seq[uint8]
    pq*, npq*: ref ProductQuantizer

type
  Vector* = object
    idata*: seq[float32]

type
  Node* = object
    parent*: int32
    left*: int32
    right*: int32
    count*: int64
    binary*: bool

  Model* = object
    rng*: Rand
    wi*: ptr Matrix
    wo*: ptr Matrix
    qwi*: ptr QMatrix
    qwo*: ptr QMatrix
    args*: ref Args
    hidden*: Vector
    output*: Vector
    grad*: Vector
    hsz*: int32
    osz*: int32
    loss*: float32
    nexamples*: int64
    quant*: bool
    t_sigmoid*: Vector
    t_log*: Vector
    negatives*: seq[int32]
    negpos*: uint32
    paths*: seq[seq[int32]]
    codes*: seq[seq[bool]]
    tree*: seq[Node]

type
  FastText* = ref object
    args*: ref Args
    dict*: ref Dictionary
    input*: Matrix
    output*: Matrix
    qinput*: QMatrix
    qoutput*: QMatrix
    model*: ref Model
    quant*: bool
    version*: int32

proc size*(self: Matrix; dim: int64): int64 =
  assert(dim == 0 or dim == 1)
  result = if dim == 0: self.m else: self.n



proc initVector*(a1: int64): Vector =
  result.idata = newSeq[float32](a1)

proc initVector*(a1: Vector): Vector =
  result = a1

proc size*(self: Vector): int64 =
  self.idata.len

proc `[]`*(self: var Vector; i: int64): ptr float32 =
  result = self.idata[i].addr

proc `[]`*(self: Vector; i: int64): ptr float32 =
  result = self.idata[i].unsafeAddr

proc `[]`*(self: ptr float32; key: int64): ptr float32 =
  let a: ptr UncheckedArray[float32] = cast[ptr UncheckedArray[float32]](self)
  a[key].unsafeaddr

proc `[]`*(self: ptr uint8; key: int64): ptr uint8 =
  let a: ptr UncheckedArray[uint8] = cast[ptr UncheckedArray[uint8]](self)
  a[key].unsafeaddr

proc get*(self: var Vector; i: int64): float32 =
  self.idata[i]

proc at*(self: Matrix; i: int64; j: int64): float32 {.noSideEffect.} =
  self.idata[ (i * self.n + j)]

proc at*(self: var Matrix; i: int64; j: int64): ptr float32 =
  self.idata[ (i * self.n + j)].unsafeAddr

proc rows*(self: Matrix): int64 =
  self.m

proc cols*(self: Matrix): int64 =
  self.n

proc dotRow*(self: Matrix; vec: var Vector;
  i: int32): float32 {.noSideEffect.} =
  assert i >= 0
  assert i < self.m
  assert vec.size == self.n
  for j in 0..<self.n:
    result += self.at(i, j) * vec.get(j.int64)
  if classify(result) == math.fcNan:
    raise newException(ValueError, "Encountered NaN.")

proc getM*(self: QMatrix): int64 =
  self.m

proc getN*(self: QMatrix): int64 =
  self.n

proc get_centroids*(self: var ProductQuantizer; m: int32;
  i: uint8): ptr float32 =
  if (m == self.nsubq - 1):
    return self.centroids[m * ksub * self.dsub + i.int32 *
        self.lastdsub].addr
  return self.centroids[(m * ksub + i.int32) * self.dsub].addr

proc mulcode*(self: var ProductQuantizer; x: var Vector; codes: ptr uint8;
  t: int32; alpha: float32): float32 =
  var d = self.dsub
  let code = codes[self.nsubq * t]
  var c: ptr float32
  for m in 0..<self.nsubq:
    c = self.get_centroids(m.int32, code[m][])
    if m == self.nsubq - 1:
      d = self.lastdsub
    for n in 0..<d:
      result += x[m * self.dsub + n][] * c[n][]
  result = result * alpha

proc addcode*(self: var ProductQuantizer; x: var Vector; codes: ptr uint8;
  t: int32; alpha: float32) =
  var d = self.dsub
  let code = codes[self.nsubq * t]
  var c: ptr float32
  for m in 0..<self.nsubq:
    c = self.get_centroids(m.int32, code[m][])
    if m == self.nsubq - 1:
      d = self.lastdsub
    for n in 0..<d:
      x[m * self.dsub + n][] += alpha * c[n][]

proc addToVector*(self: var QMatrix; x: var Vector; t: int32) =
  var norm: float32 = 1
  if self.qnorm:
    norm = self.npq[].get_centroids(0'i32, self.norm_codes[t])[]
  self.pq[].addcode(x, self.codes[0].addr, t, norm)

proc dotRow*(self: var QMatrix; vec: var Vector; i: int64): float32 =
  assert(i >= 0);
  assert(i < self.m)
  assert(vec.size() == self.n)
  var norm: float32 = 1
  if self.qnorm:
    norm = self.npq[].get_centroids(0'i32, self.norm_codes[i])[]
  self.pq[].mulcode(vec, self.codes[0].addr, i.int32, norm)

proc l2NormRow*(self: var Matrix; i: int64): float32 {.noSideEffect.} =
  var norm: float32 = 0.0
  for j in 0..<self.n:
    norm += self.at(i, j)[]

  if norm == NaN:
    raise newException(ValueError, "Encountered NaN.")
  sqrt(norm)

proc l2NormRow*(self: var Matrix; norms: var Vector) {.noSideEffect.} =
  assert norms.size == self.m
  for i in 0..<self.m:
    norms[i][] = self.l2NormRow(i)

proc addRow*(self: var Matrix; vec: var Vector; i: int64; a: float32) =
  assert i >= 0
  assert i < self.m
  assert vec.size == self.n
  for j in 0..<self.n:
    self.idata[ (i * self.n + j).int32] += a * vec.get(j)

proc multiplyRow*(self: var Matrix; nums: var Vector; ib: int64 = 0;
  ie: int64 = -1) =
  var iee = ie
  if ie == -1:
    iee = self.m
  assert iee <= nums.size
  var i = ib
  var n: float32
  while i < iee:
    n = nums.get(i - ib)
    if n != 0:
      for j in 0..<self.n:
        self.at(i, j)[] *= n
    inc i

proc divideRow*(self: var Matrix; denoms: var Vector; ib: int64 = 0;
  ie: int64 = -1) =
  var iee = ie
  if ie == -1:
    iee = self.m
  assert iee <= denoms.size
  var i = ib
  var n: float32
  while i < iee:
    n = denoms.get(i - ib)
    if n != 0:
      for j in 0..<self.n:
        self.at(i, j)[] /= n
    inc i

proc newArgs*(): ref Args =
  result = new Args
  result.lr = 0.05
  result.dim = 100
  result.ws = 5
  result.epoch = 5
  result.minCount = 5
  result.minCountLabel = 0
  result.neg = 5
  result.wordNgrams = 1
  result.loss = loss_name.ns
  result.model = model_name.sg
  result.bucket = 2000000
  result.minn = 3
  result.maxn = 6
  result.thread = 12
  result.lrUpdateRate = 100
  result.t = 1e-4
  result.label = "__label__"
  result.verbose = 2
  result.pretrainedVectors = ""
  result.saveOutput = false

  result.qout = false
  result.retrain = false
  result.qnorm = false
  result.cutoff = 0
  result.dsub = 2


