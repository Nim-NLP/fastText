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
  entry_type* = enum
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
    args:ptr Args
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

proc initDictionary*(a1: ptr Args): Dictionary =
    result.args = a1
    let i:int32 = -1
    result.word2int = newSeq[i](MAX_VOCAB_SIZE)
    result.pruneidxsize = -1'i32
    result.pruneidx =  initTable[int32,int32]()

{.this: self.}
proc load*(self: var Dictionary; a2: var Stream)

proc initDictionary*(a1: ptr Args,stream:var Stream): Dictionary =
    result.args = a1
    let i:int32 = -1
    result.word2int = newSeq[i](MAX_VOCAB_SIZE)
    result.pruneidxsize = -1
    result.pruneidx =  initTable[int32,int32]()
    result.load(stream)

{.this: self.}
proc initTableDiscard(self:var Dictionary) = 
    pdiscard.setLen(size);
    
    var 
        i = 0
        f:float32
    while i < size:
        f = float32(words[i].count) / float32(ntokens);
        pdiscard[i] = sqrt(args.t / f) + args.t / f;
        inc i

{.this: self.}
proc pushHash(self: Dictionary;hashes:var seq[int32] ,id:int32) {.noSideEffect.} =
    if (pruneidxsize == 0 or id < 0):
        return 
    var tid = id
    if (pruneidxsize > 0) :
        if (pruneidx.hasKey(id)):
            tid = pruneidx[id];
        else:
            return;
    hashes.add(nwords + tid);

proc hash*(this: Dictionary; str: string): uint32 {.noSideEffect.} =
    var 
        h:uint32 = 2166136261'u32
        i = 0
    while i < str.len():
        h = h ^ cast[uint32](cast[int8](str[i]));
        h = h * 16777619;
        inc i
    return h;
        
{.this: self.}
proc computeSubwords*(self: Dictionary; word: string; ngrams: var seq[int32];
                     substrings: ptr seq[string] = nil) {.noSideEffect.} =
    var 
        i = 0
        j:int
        n:int
        h:int32
        ngram:string
        c:uint8
    # debugEcho word,word.len()
    while i < word.len():
        c = (cast[uint8](word[i]) and 0xC0)
        # debugEcho c == 0x80
        if (c == 0x80): 
            inc i
            continue 
        else: 
            discard
        # debugEcho 111
        j = i
        n = 1
        while ( j < word.len() and n <= args.maxn):
            ngram.add(word[j])
            inc j
            while (j < word.len() and (cast[uint8](word[j]) and 0xC0) == 0x80):
                ngram.add(word[j])
                inc j
            
            # debugEcho "n is",n
            if (n >= args.minn and not (n == 1 and (i == 0 or j == word.len()))):
                h = (int32) self.hash(ngram) mod cast[uint32](args.bucket)
                self.pushHash(ngrams, h)
                if (substrings != nil):
                    substrings[].add(ngram)
            inc n
        inc i
        # debugEcho "i is",i

{.this: self.}
proc initNgrams(self:var Dictionary) =
    var 
        i = 0'i32
        word:string
    while i < size:
        word = BOW & words[i].word & EOW;
        words[i].subwords.setLen(0);
        words[i].subwords.add(i);
        if (words[i].word != EOS):
            self.computeSubwords(word, words[i].subwords)
        inc i

{.this: self.}
proc find(self: Dictionary,w:string,  h:uint32):int32 {.noSideEffect.} =
    var 
        word2intsize:uint32 = word2int.len().uint32
        id = h mod word2intsize
    debugEcho w
    while (word2int[id] != -1'i32 and words[word2int[id]].word != w):
      id = (id + 1) mod word2intsize;
    #   debugEcho id
    return id.int32
  
proc find(self: Dictionary,w:string):int32 {.noSideEffect.} = 
    return self.find(w, self.hash(w));

{.this: self.}
proc load*(self: var Dictionary; a2: var Stream) =
    words.setLen(0)
    discard a2.readData(addr size,sizeof(int32))
    discard a2.readData(addr nwords,sizeof(int32))
    discard a2.readData(addr nlabels,sizeof(int32))
    discard a2.readData(addr ntokens,sizeof(int64))
    discard a2.readData(addr pruneidxsize,sizeof(int64))
    var
        c:char
        e:entry
        i = 0

    while i < size:
        var s: seq[char]
        while (c = a2.readChar();c.uint != 0):
            s.add(c)
        discard a2.readData(addr e.count,sizeof(int64))
        discard a2.readData(addr e.entry_type,sizeof(int32))
        words.add(e)
        inc i
    debugEcho "readData"
    pruneidx.clear();
    i = 0
    var 
        first:int32
        second:int32
    while i < pruneidxsize:
        discard a2.readData(addr first,sizeof(int32))
        discard a2.readData(addr second,sizeof(int32))
        pruneidx[first] = second
        inc i
    debugEcho "initTableDiscard"
    self.initTableDiscard()
    debugEcho  "initTableDiscard finished"
    debugEcho  "initNgrams"
    self.initNgrams()
    debugEcho  "initNgrams finished"
    let word2intsize = ceil(size.toFloat() / 0.7).toInt
    word2int[word2intsize] = -1'i32
    var j = 0'i32
    while j < size:
        word2int[self.find(words[j].word)] = j;
        # debugEcho "word2int",j
        inc j

    debugEcho "load finished"

# proc nwords*(this: Dictionary): int32 {.noSideEffect, stdcall, importcpp: "nwords",
#                                     header: headerdictionary.}
# proc nlabels*(this: Dictionary): int32 {.noSideEffect, stdcall, importcpp: "nlabels",
#                                      header: headerdictionary.}
# proc ntokens*(this: Dictionary): int64 {.noSideEffect, stdcall, importcpp: "ntokens",
#                                      header: headerdictionary.}
proc getId*(self: Dictionary; w: string): int32 {.noSideEffect.} =
    let h = self.find(w)
    return self.word2int[h];

proc getId*(self: Dictionary; w: string; h: uint32): int32 {.noSideEffect.} =
    let id = self.find(w, h);
    return self.word2int[id];

proc getType*(self: Dictionary; id: int32): entry_type {.noSideEffect.} =
    doAssert(id >= 0);
    doAssert(id < self.size);
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

# proc skipUntil(s: string; until: string; unless = '\0'; start: int): int =
#     # Skips all characters until the string `until` is found. Returns 0
#     # if the char `unless` is found first or the end is reached.
#     var i = start
#     var u = 0
#     while true:
#       if i >= s.len or s[i] == unless:
#         return 0
#       elif s[i] == until[0]:
#         u = 1
#         while i+u < s.len and u < until.len and s[i+u] == until[u]:
#           inc u
#         if u >= until.len: break
#       inc(i)
#     result = i+u-start

proc readWord*(self: Dictionary; i:  Stream; word: var string): bool  =
    var idx, old = 0
    for line in i.lines:
        while idx < line.len:
            old = idx
            if scanp(line, idx, *(~ {' ', '\n', '\r','\t','\v' ,'\f','\0',}) -> word.add($_ & EOS),  $index):
                idx = old + 1

# proc readFromFile*(this: var Dictionary; a2: var istream) {.stdcall,
#     importcpp: "readFromFile", header: headerdictionary.}
proc getLabel*(self: Dictionary; lid: int32): string {.noSideEffect.} =
    if lid < 0 or lid >= self.nlabels:
        raise newException(ValueError,"Label id is out of range [0, " & self.nlabels.intToStr)
    return self.words[lid + self.nwords].word
# proc save*(this: Dictionary; a2: var ostream) {.noSideEffect, stdcall,
#     importcpp: "save", header: headerdictionary.}
proc getCounts*(self: Dictionary; a2: entry_type): seq[int64] {.noSideEffect.} =
    for w in self.words:
        if w.entry_type == a2:
            result.add(w.count)

proc addSubwords*(self:Dictionary; line:var seq[int32]; token:string; wid:int32) =
    if wid < 0:
        if token != EOS:
            computeSubwords(BOW & token & EOW, line);
    else:
        if self.args.maxn <= 0:
            line.add(wid)
        else:
            let ngrams = self.getSubwords(wid)
            line = ngrams

proc reset*(self:Dictionary;i:Stream) =
    i.setPosition(0)
    if i.atEnd():
        i.close()
        i.setPosition(0)


proc getLine*(self: Dictionary; i:  Stream; words: var seq[int32];
             labels: var seq[int32]): int32  =
    var word_hashes:seq[int32]
    var token:string
    var ntokens = 0
    self.reset(i)
    words.setLen(0)
    labels.setLen(0)
    var h:uint32
    var wid:int32
    var typ:entry_type
    while self.readWord(i,token):
        h = hash(token)
        wid = getId(token,h)
        typ = if wid < 0 : getType(token) else: getType(wid)
        inc ntokens
        if typ == entry_type.word:
            addSubwords(words,token,wid)
            word_hashes.add(h.int32)
        elif typ == entry_type.label and wid >= 0:
            labels.add(wid - self.nwords)
        if token == EOS:
            break

proc getLine*(self: Dictionary; i:  Stream; words: var seq[int32];rng: var Rand): int32  =
    var token:string
    var ntokens:int32 = 0
    # reset(in);
    words.setLen(0)
    var h,wid:int32
    while readWord(i,token):
        h = find(token)
        wid = self.word2int[h]
        if wid < 0:continue
        inc ntokens
        if getType(wid) == entry_type.word and  self.discard(wid,rng.rand(1.0)) == false:
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

{.this: self.}
proc init*(self: var Dictionary) =
    initTableDiscard()
    initNgrams()

