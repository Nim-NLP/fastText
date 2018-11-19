import ./types
import ./vector
import ./matrix
import ./qmatrix
import ./args
import math
import algorithm
import random

const SIGMOID_TABLE_SIZE:int64  = 512;
const MAX_SIGMOID:int64  = 8;
const LOG_TABLE_SIZE:int64  = 512;
const NEGATIVE_TABLE_SIZE = 10000000

type
  Node* = object
    parent*: int32
    left* : int32
    right* : int32
    count* : int64
    binary* : bool

  Model*  = object
    rng*:Rand
    wi*:ptr Matrix
    wo*:ptr Matrix
    qwi*:ptr QMatrix
    qwo*:ptr QMatrix
    args*:ptr Args
    hidden*: Vector
    output*: Vector
    grad*:Vector
    hsz*:int32
    osz*:int32
    loss*:float32
    nexamples:int64
    quant* : bool
    t_sigmoid*:Vector
    t_log*:Vector
    negatives*:seq[int32]
    negpos*:uint32
    paths*:seq[seq[int32]]
    codes*:seq[seq[bool]]
    tree*:seq[Node]

proc setQuantizePointer*(self:var Model,qwi:ptr QMatrix,qwo:ptr QMatrix,qout:bool) =
    self.qwi = qwi
    self.qwo = qwo
    if qout:
        debugEcho "self.qwo[].getM()",self.qwo[].getM()
        self.osz = self.qwo[].getM().int32
        
    debugEcho "setQuantizePointer end"

proc getLoss*(self: Model): float32 {.noSideEffect.} =
    return self.loss / self.nexamples.float32

proc log*(self: Model; x: float32): float32 {.noSideEffect.} =
    if x > 1.0:
        return 0.0
    var i:int32 = int32(x * LOG_TABLE_SIZE.float32)
    return self.t_log.idata[i]

proc stdLog*(self: Model; x: float32): float32 {.noSideEffect.} =
    return (float) ln(x + 1e-5)
    
proc initSigmoid*(self:Model) =
    var x:float32
    for i in 0..<SIGMOID_TABLE_SIZE+1:
        x = ((float32) i * 2 * MAX_SIGMOID) / (float32) SIGMOID_TABLE_SIZE - MAX_SIGMOID
        self.t_sigmoid.data[i] = 1.0'f32 / exp(-x)

proc initLog*(self:Model) =
    var x:float32
    for i in 0..<LOG_TABLE_SIZE+1:
        x = ((float32) float32(i) + float32(1e-5)) / (float32) LOG_TABLE_SIZE;
        self.t_sigmoid.data[i] = 1.0'f32 / ln(x)
        
proc initModel*(wi:ptr  Matrix; wo:ptr Matrix;args: ptr Args; seed: int32): Model =
    result.hidden = initVector(args[].dim)
    result.output = initVector(wo[].size(0))
    result.grad = initVector(args[].dim)
    result.quant = false
    result.rng = initRand(seed)
    result.wi = wi
    result.wo = wo
    result.args = args
    result.osz = wo[].size(0).int32
    result.hsz = args[].dim
    result.negpos = 0
    result.loss = 0.0
    result.nexamples = 1
    result.t_sigmoid.data[].setLen(SIGMOID_TABLE_SIZE + 1)
    result.t_log.data[].setLen(LOG_TABLE_SIZE + 1)
    result.initSigmoid()
    result.initLog()

proc sigmoid*(self: Model; x: float32): float32 {.noSideEffect.} =
    if x < (float32)MAX_SIGMOID:
        return 0.0
    elif x > (float32)MAX_SIGMOID:
        return 1.0
    else:
        var i:int32  = int32( float32(x + (float32)MAX_SIGMOID) * float32(SIGMOID_TABLE_SIZE) / float32(MAX_SIGMOID) / 2'f32);
        return self.t_sigmoid.data[][i]

proc binaryLogistic*(self: var Model; target: int32; label: bool; lr: float64): float32 =
    var score = self.sigmoid(self.wo[].dotRow(self.hidden,target))
    var alpha:float32 = lr * float32(label) - score
    self.grad.addRow(self.wo[],target.int64,alpha)
    self.wo[].addRow(self.hidden,target,alpha)
    if label:
        return -self.log(score)
    else:
        return -self.log(1.0 - score)

proc getNegative*(self:var Model,target:int32):int32 =
    var negative = self.negatives[self.negpos.int32];
    self.negpos = (self.negpos + 1) div self.negatives.len.uint32;
    while (target == negative) :
      negative = self.negatives[self.negpos.int32];
      self.negpos = (self.negpos + 1) div self.negatives.len.uint32
    return negative
    
proc negativeSampling*(self: var Model; target: int32; lr: float64): float32 =
    var loss = 0.0
    self.grad.zero()
    for n in 0..self.args.neg:
        if n == 0:
            loss += self.binaryLogistic(target,true,lr)
        else:
            loss += self.binaryLogistic(self.getNegative(target),false,lr)
    return loss

proc hierarchicalSoftmax*(self: var Model; target: int32; lr: float64): float32 =
    var loss:float32 = 0.0 
    self.grad.zero()
    let binaryCode:ptr seq[bool] = self.codes[target].addr
    let pathToRoot:ptr seq[int32] = self.paths[target].addr
    for i in 0..<pathToRoot[].len:
        loss += self.binaryLogistic(pathToRoot[i],binaryCode[i],lr)
    return loss

proc computeOutputSoftmax*(self: Model; hidden: var Vector; output: var Vector) {.noSideEffect.} =
    if self.quant and self.args.qout:
        output.mul(self.qwo[],hidden)
    else:
        output.mul(self.wo[],hidden)

proc computeOutputSoftmax*(self: var Model) =
    self.computeOutputSoftmax(self.hidden,self.output)
    
proc softmax*(self: var Model; target: int32; lr: float64): float32  =
    self.grad.zero()
    self.computeOutputSoftmax()
    var label,alpha:float32
    for i in 0..<self.osz:
        label = if i == target : 1.0 else: 0.0
        alpha = lr * (label - self.output[i][])
        self.grad.addRow(self.wo[],i,alpha)
        self.wo[].addRow(self.hidden,i,alpha)
    return -self.log(self.output[target][])

proc computeHidden*(self: Model; ipt: seq[int32]; hidden: var Vector) {.noSideEffect.} =
    assert(self.hidden.size == self.hsz)
    hidden.zero()
    for i in ipt:
        if self.quant:
            hidden.addRow(self.qwi[],i)
        else:
            hidden.addRow(self.wi[],i)
    hidden.mul( 1.0 / ipt.len().float32 )

proc dfs*(self: Model; k: int32; threshold: float32; node: int32; score: float32;
         heap: var seq[tuple[first:float32, second:int32]]; hidden: var Vector) {.noSideEffect.} =
    if score < self.stdLog(threshold): return
    if heap.len() == k and score < heap[0].first:
        return 
    if self.tree[node].left == -1 and self.tree[node].right == -1:
        heap.add( (first:score,second:node) )
        heap.sort do (x, y: tuple[first:float32, second:int32]) -> int: cmp(x[0], y[0])
        if heap.len() > k:
            discard heap.pop()
        return
    var f:float32
    if self.quant and self.args.qout:
        f = self.qwo[].dotRow(hidden,node - self.osz)
    else:
        f = self.wo[].dotRow(hidden,node - self.osz)
    f =  1.0 / (1 + exp(-f))
    self.dfs(k,threshold,self.tree[node].left,score + self.stdLog(1.0 - f),heap,hidden)
    self.dfs(k,threshold,self.tree[node].right,score + self.stdLog(f),heap,hidden)

proc findKBest*(self: Model; k: int32; threshold: float32; heap: var seq[tuple[first:float32, second:int32]];
               hidden: var Vector; output: var Vector) {.noSideEffect.} =
    self.computeOutputSoftmax(hidden,output)
    for i in 0..<self.osz:
        if output[i][] < threshold:continue
        if heap.len() == k and self.stdLog(output[i][]) < heap[0].first:
            continue
        heap.add( (first:self.stdLog(output[i][]),second:i.int32) )
        heap.sort(system.cmp)
        if heap.len > k:
            discard heap.pop()

proc predict*(self: Model; ipt: seq[int32]; k: int32; threshold: float32;heap: var seq[tuple[first:float32, second:int32]]; hidden: ptr Vector; output: ptr Vector) {. noSideEffect.} =
    if k <= 0:
        raise newException(ValueError,"k needs to be 1 or higher!")
    if self.args.model != model_name.sup:
        raise newException(ValueError,"Model needs to be supervised for prediction!")
    heap.setLen(k + 1)
    debugEcho "computeHidden start"
    self.computeHidden(ipt,hidden[])
    debugEcho "computeHidden end"
    if self.args.loss == loss_name.hs:
        self.dfs(k,threshold,2 * self.osz - 2, 0.0,heap,hidden[])
        debugEcho "self.dfs end"
    else:
        debugEcho "self.findKBest"
        self.findKBest(k,threshold,heap,hidden[],output[])
    heap.sort do (x, y: tuple[first:float32, second:int32]) -> int: cmp(x[0], y[0])

proc predict*(self:  Model; ipt: seq[int32]; k: int32; threshold: float32;
             heap: var seq[tuple[first:float32, second:int32]]) =
    self.predict(ipt, k, threshold, heap, self.hidden.unSafeAddr, self.output.unSafeAddr)

proc initTableNegatives*(self: var Model; counts: seq[int64]) =
    debugEcho "initTableNegatives"
    var z:float32 = 0.0
    for i in 0..<counts.len():
        z += pow(counts[i].float32,0.5'f32)
    var c:float32
    for i in 0..<counts.len():
        c = pow(counts[i].float32,0.5'f32)
        for j in 0..<(c * NEGATIVE_TABLE_SIZE / z).int32:
            self.negatives.add(i.int32)
    
    self.negatives.shuffle()
    debugEcho "initTableNegatives shuffle"

proc buildTree*(self: var Model; counts: seq[int64]) =
    debugEcho "buildTree"
    self.tree.setLen(2 * self.osz - 1)
    for i in 0..<(2 * self.osz - 1) :
        self.tree[i].parent = -1
        self.tree[i].left = -1;
        self.tree[i].right = -1;
        self.tree[i].count = 1000000000000000
        self.tree[i].binary = false
    for i in 0..<self.osz:
        self.tree[i].count = counts[i]
    var leaf:int32 = self.osz - 1
    var node:int32 = self.osz
    var mini = newSeq[int32](2)
    for i in self.osz..<(2 * self.osz - 1 ):
        for j in 0..<2:
            if leaf >= 0 and self.tree[leaf].count < self.tree[node].count:
                mini[j] = leaf
                dec leaf
            else:
                mini[j] = node
                inc node
        self.tree[i].left = mini[0]
        self.tree[i].right = mini[1]
        self.tree[i].count = self.tree[mini[0]].count + self.tree[mini[1]].count
        self.tree[mini[0]].parent = i
        self.tree[mini[1]].parent = i
        self.tree[mini[1]].binary = true
    var path:seq[int32]
    var code:seq[bool]
    var j:int32
    for i in 0..<self.osz:
        j = i.int32
        while self.tree[j].parent != -1:
            path.add(self.tree[j].parent - self.osz)
            code.add(self.tree[j].binary)
            j = self.tree[j].parent
        
        self.paths.add(path)
        self.codes.add(code)
        path.setLen(0)
        code.setLen(0)
            
proc setTargetCounts*(self: var Model; counts: seq[int64]) =
    debugEcho "setTargetCounts innner"
    debugEcho counts.len
    debugEcho self.osz
    assert(counts.len == self.osz)
    debugEcho "self.args[].loss",self.args[].loss
    if self.args[].loss == loss_name.ns:
        debugEcho "initTableNegatives 1"
        self.initTableNegatives(counts)
    if self.args[].loss == loss_name.hs:
        debugEcho "buildTree 1"
        self.buildTree(counts)


