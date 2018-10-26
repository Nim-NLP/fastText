import tables
import sequtils
import math
import ./args
# import ./real
import streams

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
    nwords:int32
    nlabels:int32
    ntokens:int64
    pruneidxsize:int64
    pruneidx:Table[int32,int32]
    
proc isPruned*(self:var Dictionary):bool =
    self.pruneidx_size >= 0

proc initDictionary*(a1: ptr Args): Dictionary =
    result.args = a1
    let i:int32 = -1
    result.word2int = newSeq[i](MAX_VOCAB_SIZE)
    result.pruneidxsize = -1'i32

{.this: self.}
proc load*(self: var Dictionary; a2: var Stream)

proc initDictionary*(a1: ptr Args,stream:var Stream): Dictionary =
    result.args = a1
    let i:int32 = -1
    result.word2int = newSeq[i](MAX_VOCAB_SIZE)
    result.pruneidxsize = -1
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
    while i < word.len():
        if ((cast[int](word[i]) and 0xC0) == 0x80): continue else: discard
        j = i
        n = 1
        while ( j < word.len() and n <= args.maxn):
            ngram.add(word[j]);
            inc j
            while (j < word.len() and (cast[int](word[j]) and 0xC0) == 0x80):
                ngram.add(word[j]);
                inc j
            inc n
        
        if (n >= args.minn and not (n == 1 and (i == 0 or j == word.len()))) :
            h = (int32) self.hash(ngram) mod cast[uint32](args.bucket);
            self.pushHash(ngrams, h);
            if (substrings != nil):
                substrings[].add(ngram);
        inc i

proc initNgrams(self:var Dictionary) =
    var 
        i = 0'i32
        word:string

    while i < size:
        word = BOW & words[i].word & EOW;
        words[i].subwords.setLen(0);
        words[i].subwords.add(i);
        if (words[i].word != EOS) :
            self.computeSubwords(word, words[i].subwords);
        inc i
  


{.this: self.}
proc find(self:var Dictionary,w:string,  h:uint32):int32 {.noSideEffect.} =
    var 
        word2intsize:uint32 = cast[uint32](word2int.len())
        id = cast[int](h mod word2intsize)
    while (word2int[id] != -1 and words[word2int[id]].word != w) :
      id = (id + 1) mod cast[int](word2intsize);
    return cast[int32](id)
  
proc find(self:var Dictionary,w:string):int32 {.noSideEffect.} = 
    return self.find(w, self.hash(w));

{.this: self.}
proc load*(self: var Dictionary; a2: var Stream) =
    words.setLen(0)
    discard a2.readData(addr size,sizeof(size))
    discard a2.readData(addr nwords,sizeof(nwords))
    discard a2.readData(addr nlabels,sizeof(nlabels))
    discard a2.readData(addr ntokens,sizeof(ntokens))
    discard a2.readData(addr pruneidxsize,sizeof(pruneidxsize))
    var
        c:char
        e:entry
        i = 0
    while i < size:
        while (c = a2.peekChar();c !=  '0'):
            e.word.add(c)
        discard a2.readData(addr e.count,sizeof(e.count))
        discard a2.readData(addr e.entry_type,sizeof(entry_type))
        words.add(e);
        inc i
    
    pruneidx.clear();
    i = 0
    var 
        first:int32
        second:int32
    while i < pruneidxsize:
        discard a2.readData(addr first,sizeof(first))
        discard a2.readData(addr second,sizeof(second))
        pruneidx[first] = second;
        inc i
    
    self.initTableDiscard()
    self.initNgrams()

    let word2intsize = ceil(size.toFloat() / 0.7).toInt
    word2int[word2intsize] = -1'i32
    var j = 0'i32
    while j < size:
        word2int[self.find(words[j].word)] = j;
        inc j
        
proc constructDictionary*(a1: ptr Args; a2: var Stream): Dictionary =
    result.pruneidxsize = -1
    result.load(a2);

# proc nwords*(this: Dictionary): int32 {.noSideEffect, stdcall, importcpp: "nwords",
#                                     header: headerdictionary.}
# proc nlabels*(this: Dictionary): int32 {.noSideEffect, stdcall, importcpp: "nlabels",
#                                      header: headerdictionary.}
# proc ntokens*(this: Dictionary): int64 {.noSideEffect, stdcall, importcpp: "ntokens",
#                                      header: headerdictionary.}
# proc getId*(this: Dictionary; a2: string): int32 {.noSideEffect, stdcall,
#     importcpp: "getId", header: headerdictionary.}
# proc getId*(this: Dictionary; a2: string; h: uint32): int32 {.noSideEffect, stdcall,
#     importcpp: "getId", header: headerdictionary.}
# proc getType*(this: Dictionary; a2: int32): entry_type {.noSideEffect, stdcall,
#     importcpp: "getType", header: headerdictionary.}
# proc getType*(this: Dictionary; a2: string): entry_type {.noSideEffect, stdcall,
#     importcpp: "getType", header: headerdictionary.}
# proc `discard`*(this: Dictionary; a2: int32; a3: real): bool {.noSideEffect, stdcall,
#     importcpp: "discard", header: headerdictionary.}
# proc getWord*(this: Dictionary; a2: int32): string {.noSideEffect, stdcall,
#     importcpp: "getWord", header: headerdictionary.}
# proc getSubwords*(this: Dictionary; a2: int32): vector[int32] {.noSideEffect, stdcall,
#     importcpp: "getSubwords", header: headerdictionary.}
# proc getSubwords*(this: Dictionary; a2: string): vector[int32] {.noSideEffect, stdcall,
#     importcpp: "getSubwords", header: headerdictionary.}

# proc getSubwords*(this: Dictionary; a2: string; a3: var vector[int32];
#                  a4: var vector[string]) {.noSideEffect, stdcall,
#                                         importcpp: "getSubwords",
#                                         header: headerdictionary.}



# proc add*(this: var Dictionary; a2: string) {.stdcall, importcpp: "add",
#                                         header: headerdictionary.}
# proc readWord*(this: Dictionary; a2: var istream; a3: var string): bool {.noSideEffect,
#     stdcall, importcpp: "readWord", header: headerdictionary.}
# proc readFromFile*(this: var Dictionary; a2: var istream) {.stdcall,
#     importcpp: "readFromFile", header: headerdictionary.}
# proc getLabel*(this: Dictionary; a2: int32): string {.noSideEffect, stdcall,
#     importcpp: "getLabel", header: headerdictionary.}
# proc save*(this: Dictionary; a2: var ostream) {.noSideEffect, stdcall,
#     importcpp: "save", header: headerdictionary.}

    
    
# proc getCounts*(this: Dictionary; a2: entry_type): vector[int64] {.noSideEffect,
#     stdcall, importcpp: "getCounts", header: headerdictionary.}
# proc getLine*(this: Dictionary; a2: var istream; a3: var vector[int32];
#              a4: var vector[int32]): int32 {.noSideEffect, stdcall,
#     importcpp: "getLine", header: headerdictionary.}
# proc getLine*(this: Dictionary; a2: var istream; a3: var vector[int32];
#              a4: var minstd_rand): int32 {.noSideEffect, stdcall,
#                                        importcpp: "getLine",
#                                        header: headerdictionary.}
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

