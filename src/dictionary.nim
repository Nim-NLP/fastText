import tables
import sequtils
import math
import ./args
# import ./real
import streams
import strutils
import random
import strscans

type
  id_type* = int32
  entry_type* = enum # enum class entry_type : int8_t {word=0, label=1};
    word = 0, label = 1

const EOS* = "</s>";
const BOW* = "<";
const EOW* = ">";
const MAX_VOCAB_SIZE:int32 = 30000000
const MAX_LINE_SIZE:int32 = 1024

type
  entry*  = object
    word* : string
    count* : int64
    entry_type* : entry_type
    subwords* : seq[int32]

  Dictionary*  = object
    args: Args
    word2int:seq[int32]
    words:seq[entry] 
    pdiscard:seq[float32]
    size:int32
    nwords*:int32
    nlabels*:int32
    ntokens*:int64
    pruneidxsize:int64
    pruneidx:Table[int32,int32]
    
proc isPruned*(self:var Dictionary):bool =
    self.pruneidx_size >= 0

# proc initDictionary*(a1: ptr Args): Dictionary =
#     result.args = a1
#     let i:int32 = -1
#     result.word2int = newSeq[i](MAX_VOCAB_SIZE)
#     result.pruneidxsize = -1'i32
#     result.pruneidx =  initTable[int32,int32]()

proc load*(self: var Dictionary; a2: var Stream)

proc initDictionary*(a1: Args,stream:var Stream): Dictionary =
    result.args = a1
    let i:int32 = -1
    result.word2int = newSeq[i](MAX_VOCAB_SIZE)
    result.pruneidxsize = -1
    result.pruneidx =  initTable[int32,int32]()
    result.load(stream)

proc initTableDiscard(self:var Dictionary) = 
    self.pdiscard.setLen(self.size)
    var 
        i = 0
        f:float32
    while i < self.size:
        f = float32(self.words[i].count) / float32(self.ntokens);
        self.pdiscard[i] = sqrt(self.args.t / f) + self.args.t / f
        inc i

proc pushHash(self: Dictionary;hashes:var seq[int32] ,id:int32) {.noSideEffect.} =
    if (self.pruneidxsize == 0 or id < 0):
        return 
    var tid = id
    if (self.pruneidxsize > 0):
        if (self.pruneidx.hasKey(id)):
            tid = self.pruneidx[id]
        else:
            return
    hashes.add(self.nwords + tid);

proc hash*(self: Dictionary; str: string): uint32 {.noSideEffect.} =
    var 
        h:uint32 = 2166136261'u32
        i = 0
    for i in 0..<str.len():
        h = h ^ cast[uint32](cast[int8](str[i]));
        h = h * 16777619
    return h

proc computeSubwords*(self: Dictionary; word: string; ngrams: var seq[int32];
                     substrings: ptr seq[string] = nil) {.noSideEffect.} =
    var 
        i = 0
        j:int
        n:int
        h:int32
        ngram:string
        c:uint8
    for i in 0..<word.len():
        ngram.setLen(0)
        if ( (uint8(word[i]) and 0xC0) == 0x80): continue
        j = i
        n = 1
        while ( j < word.len() and n <= self.args.maxn):
            ngram.add(word[j])
            inc j
            while (j < word.len() and (uint8(word[j]) and 0xC0) == 0x80):
                ngram.add(word[j])
                inc j
            if (n >= self.args.minn and not (n == 1 and (i == 0 or j == word.len()))):
                h =(int32) self.hash(ngram) mod (self.args.bucket).uint32
                self.pushHash(ngrams, h)
                # debugEcho ngram
                if (substrings != nil):
                    substrings[].add(ngram)
            inc n


proc initNgrams(self:var Dictionary) =
    var 
        word:string
    for i in 0..self.size:
        word = BOW & self.words[i].word & EOW
        self.words[i].subwords.setLen(0)
        self.words[i].subwords.add(i)
        if (self.words[i].word != EOS):
            self.computeSubwords(word, self.words[i].subwords)

proc find(self: Dictionary,w:string,  h:uint32):int32 {.noSideEffect.} =
    var 
        word2intsize:uint32 = self.word2int.len().uint32
        id = h mod word2intsize
    while (self.word2int[id] != -1 and self.words[self.word2int[id]].word != w):
      id = (id + 1) mod word2intsize;
    return id.int32
  
proc find(self: Dictionary,w:string):int32 {.noSideEffect.} = 
    return self.find(w, self.hash(w));

proc load*(self: var Dictionary; a2: var Stream) =
    self.words.setLen(0)
    discard a2.readData(addr self.size,sizeof(int32))
    discard a2.readData(addr self.nwords,sizeof(int32))
    discard a2.readData(addr self.nlabels,sizeof(int32))
    discard a2.readData(addr self.ntokens,sizeof(int64))
    discard a2.readData(addr self.pruneidxsize,sizeof(int64))
    var
        c:char
        e:entry
    self.words.setLen(self.size)
    for i in 0..<self.size:
        var s: seq[char]
        while (c = a2.readChar();c != '\0'):
            s.add(c)
        e.word =  cast[string](s)
        discard a2.readData(addr e.count,sizeof(int64))
        discard a2.readData(addr e.entry_type,sizeof(int8))
        self.words.add(e)
    self.pruneidx.clear();
    var 
        first:int32
        second:int32
    for i in 0..<self.pruneidxsize:
        discard a2.readData(addr first,sizeof(int32))
        discard a2.readData(addr second,sizeof(int32))
        self.pruneidx[first] = second
    self.initTableDiscard()
    self.initNgrams()
    let word2intsize = ceil(self.size.float32 / 0.7'f32).toInt
    self.word2int[word2intsize] = -1'i32
    var j = 0'i32
    while j < self.size:
        self.word2int[self.find(self.words[j].word)] = j;
        inc j

    debugEcho "load finished"

proc getId*(self: Dictionary; w: string): int32 {.noSideEffect.} =
    let h = self.find(w)
    return self.word2int[h];

proc getId*(self: Dictionary; w: string; h: uint32): int32 {.noSideEffect.} =
    let id = self.find(w, h);
    return self.word2int[id];

proc getType*(self: Dictionary; id: int32): entry_type {.noSideEffect.} =
    assert(id >= 0);
    assert(id < self.size);
    return self.words[id].entry_type

proc getType*(self: Dictionary; w: string): entry_type {.noSideEffect.} =
    if w.find(self.args.label) == 0:
        result = entry_type.label
    else:
        result = entry_type.word
    
proc `discard`*(self: Dictionary; id: int32; rand: float32): bool {.noSideEffect.} =
    assert(id >= 0)
    assert(id < self.nwords)
    if (self.args.model == model_name.sup):
        return false
    return rand > self.pdiscard[id]
# proc getWord*(this: Dictionary; a2: int32): string {.noSideEffect, stdcall,
#     importcpp: "getWord", header: headerdictionary.}
proc getSubwords*(self: Dictionary; i: int32): seq[int32] {.noSideEffect.} =
    assert(i >= 0);
    assert(i < self.nwords);
    return self.words[i].subwords;

proc getSubwords*(self: Dictionary; word: string): seq[int32] {.noSideEffect.} =
    let i = self.getId(word)
    if i >= 0:
        return self.getSubwords(i)
    var ngrams:seq[int32]
    if word != EOS:
        self.computeSubwords(BOW & word & EOW, ngrams)
    return ngrams

proc getSubwords*(self: Dictionary; word: string; ngrams: var seq[int32];substrings: var seq[string]) {.noSideEffect.} =
    var i:int32 = self.getId(word)
    ngrams.setLen(0)
    substrings.setLen(0)
    if i >= 0:
        ngrams.add(i)
        substrings.add(self.words[i].word)
    if word != EOS:
        self.computeSubwords(BOW & word & EOW, ngrams, substrings.addr)

# proc add*(this: var Dictionary; a2: string) {.stdcall, importcpp: "add",
#                                         header: headerdictionary.}
# proc readFromFile*(this: var Dictionary; a2: var istream) {.stdcall,
#     importcpp: "readFromFile", header: headerdictionary.}
proc getLabel*(self: Dictionary; lid: int32): string {.noSideEffect.} =
    if lid < 0 or lid >= self.nlabels:
        raise newException(ValueError,"Label id is out of range [0, " & self.nlabels.intToStr)
    return self.words[lid + self.nwords].word
# proc save*(this: Dictionary; a2: var ostream) {.noSideEffect, stdcall,
#     importcpp: "save", header: headerdictionary.}
proc getCounts*(self: Dictionary; a2: entry_type): seq[int64] {.noSideEffect.} =
    debugEcho "entry_type",a2
    for w in self.words:
        if w.entry_type == a2:
            result.add(w.count)
    debugEcho "getCounts end"

proc addSubwords*(self:Dictionary; line:var seq[int32]; token:string; wid:int32) =
    if wid < 0:
        if token != EOS:
            self.computeSubwords(BOW & token & EOW, line)
    else:
        debugEcho "addSubwords else"
        if self.args.maxn <= 0:
            line.add(wid)
        else:
            let ngrams = self.getSubwords(wid)
            for gram in ngrams:
                # line[^2] = gram
                line.add gram

proc reset*(self:Dictionary;i:Stream) =
    if i.atEnd():
        i.flush()
        i.setPosition(0)

proc addWordNgrams*(self:Dictionary;line:var seq[int32];hashes:seq[int32];n:int32)=
    var h:uint64
    for i in 0..<hashes.len():
        h = (uint64)hashes[i]
        for j in i + 1..<hashes.len:
            if j > i + n:
                h = h * 116049371 + hashes[j].uint64
                self.pushHash(line,(int32)h mod self.args.bucket.uint64)

proc readLineTokens*(self:Dictionary;line:string;words: var seq[string];) =
    var token:string
    for c in line:
        if (c == ' ' or c == '\n' or c == '\r' or c == '\t' or c == '\v' or
                    c == '\f' or c == '\0') :
            if token.len > 0:
                words.add(token)
                token.setLen(0)
        else:
            token.add(c)
    if token.len != 0:
        words.add(token)
    words.add(EOS)
    
proc getLine*(self: Dictionary; i:  Stream; words: var seq[int32];
             labels: var seq[int32]): int32  =
    var word_hashes:seq[int32] = @[]
    var token:string
    var ntokens:int32 = 0
    self.reset(i)
    words.setLen(0)
    labels.setLen(0)
    var h:uint32
    var wid:int32
    var typ:entry_type
    var tokens:seq[string]
    var line:string
    discard i.readLine(line)
    self.readLineTokens(line,tokens)
    for token in tokens:
        h = self.hash(token)
        wid = self.getId(token,h)
        typ = if wid < 0 : self.getType(token) else: self.getType(wid)
        inc ntokens
        debugEcho ntokens,typ
        if typ == entry_type.word:
            self.addSubwords(words,token,wid)
            word_hashes.add(cast[int32](h))
            debugEcho word_hashes
        elif typ == entry_type.label and wid >= 0:
            labels.add(wid - self.nwords)
        if token == EOS:
            break
    self.addWordNgrams(words, word_hashes, self.args.wordNgrams)
    return ntokens

proc getLine*(self: Dictionary; i:  Stream; words: var seq[int32];rng: var Rand): int32  =
    var token:string
    var ntokens:int32 = 0
    self.reset(i)
    words.setLen(0)
    var h,wid:int32
    var line:string
    discard i.readLine(line)
    var tokens:seq[string]
    self.readLineTokens(line,tokens)
    for token in tokens:
        h = self.find(token)
        wid = self.word2int[h]
        if wid < 0:continue
        inc ntokens
        if self.getType(wid) == entry_type.word and  self.discard(wid,rng.rand(1.0)) == false:
            words.add(wid)
        if ntokens > MAX_LINE_SIZE or token == EOS:
            break
    return ntokens
            
# proc threshold*(this: var Dictionary; a2: int64; a3: int64) {.stdcall,
#     importcpp: "threshold", header: headerdictionary.}
# proc prune*(this: var Dictionary; a2: var vector[int32]) {.stdcall, importcpp: "prune",
#     header: headerdictionary.}
# proc isPruned*(this: var Dictionary): bool {.stdcall, importcpp: "isPruned",
#                                         header: headerdictionary.}
# proc dump*(this: Dictionary; a2: var ostream) {.noSideEffect, stdcall,
#     importcpp: "dump", header: headerdictionary.}

proc init*(self: var Dictionary) =
    self.initTableDiscard()
    self.initNgrams()

