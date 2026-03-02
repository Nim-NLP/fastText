import sequtils
import strutils
import tables
import math
import unicode
import ./dictionary
import ./types
import ./vector
import ./matrix


type
  Token* = object
    text*: string
    id*: int32
    subwords*: seq[string]
    subwordIds*: seq[int32]

  TokenizedResult* = object
    tokens*: seq[Token]
    labels*: seq[string]
    unknownTokens*: seq[string]
    input*: ptr Matrix


proc getWordVector(self: FastText; word: string): Vector =

  result = initVector(self.args.dim.int64)
  let ngrams = self.dict[].getSubwords(word)
  var count = 0
  if ngrams.len > 0:
    for ngram in ngrams:
      if ngram >= 0 and ngram < self.input.m:
        result.addRow(self.input, ngram.int64)
        inc count
    if count > 0:
      result.mul(1.0 / count.float32)

proc vectorSimilarity(v1, v2: Vector): float32 =
  if v1.size() == 0 or v2.size() == 0 or v1.size() != v2.size():
    return 0.0
  var dot: float32 = 0.0
  var norm1: float32 = 0.0
  var norm2: float32 = 0.0
  for i in 0..<v1.size():
    dot += v1.idata[i] * v2.idata[i]
    norm1 += v1.idata[i] * v1.idata[i]
    norm2 += v2.idata[i] * v2.idata[i]
  if norm1 > 0 and norm2 > 0:
    result = dot / (sqrt(norm1) * sqrt(norm2))

proc isCjkRune(r: Rune): bool =
  let code = r.int32
  result = (code >= 0x4E00 and code <= 0x9FFF) or
           (code >= 0x3400 and code <= 0x4DBF) or
           (code >= 0x20000 and code <= 0x2A6DF)

proc isPunctuationRune(r: Rune): bool =
  let punctuations = [',', '.', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}']
  if r.int32 <= 127:
    result = chr(r.int32) in punctuations
  else:
    let s = $r
    result = s in ["，", "。", "！", "？", "、", "；", "：", "（", "）", "【", "】", "《", "》", "—", "…"]

proc isDigitRune(r: Rune): bool =
  result = r.int32 >= ord('0') and r.int32 <= ord('9')

proc isLetterRune(r: Rune): bool =
  let c = r.int32
  result = (c >= ord('a') and c <= ord('z')) or (c >= ord('A') and c <= ord('Z'))

proc getRuneLen(s: string): int =
  result = s.runeLen

proc segmentText*(self: FastText; text: string): seq[string] =
  if text.len == 0:
    return @[]

  let runes = text.toRunes

  type TokenInfo = object
    isCjk: bool
    runes: seq[Rune]

  var tokens: seq[TokenInfo]
  var i = 0
  while i < runes.len:
    let r = runes[i]

    if r.isDigitRune():
      var t = TokenInfo(isCjk: false, runes: @[])
      while i < runes.len and runes[i].isDigitRune():
        t.runes.add(runes[i])
        inc i
      tokens.add(t)
    elif r.isLetterRune():
      var t = TokenInfo(isCjk: false, runes: @[])
      while i < runes.len and runes[i].isLetterRune():
        t.runes.add(runes[i])
        inc i
      tokens.add(t)
    elif r.isPunctuationRune():
      var t = TokenInfo(isCjk: false, runes: @[r])
      tokens.add(t)
      inc i
    elif r.isCjkRune():
      var t = TokenInfo(isCjk: true, runes: @[])
      while i < runes.len and runes[i].isCjkRune():
        t.runes.add(runes[i])
        inc i
      tokens.add(t)
    else:
      let s = $r
      if s != " " and s != "\t" and s != "\n" and s != "\r":
        var t = TokenInfo(isCjk: false, runes: @[r])
        tokens.add(t)
      inc i

  for token in tokens:
    if not token.isCjk:
      var s: string
      for r in token.runes:
        s.add($r)
      result.add(s)
    else:
      let cjkRunes = token.runes
      let n = cjkRunes.len
      if n == 0:
        continue

      var i = 0
      while i < n:
        var found = false
        let maxWordLen = min(6, n - i)
        for len in countdown(maxWordLen, 1):
          var candidate: string
          for j in 0..<len:
            candidate.add($cjkRunes[i + j])

          let wordId = self.dict[].getId(candidate)
          if wordId >= 0:
            result.add(candidate)
            i += len
            found = true
            break

        if not found:
          var bestLen = 1
          var bestScore = -1.0

          for len in 1..min(3, n - i):
            var candidate: string
            for j in 0..<len:
              candidate.add($cjkRunes[i + j])

            let vec = self.getWordVector(candidate)
            let score = vec.norm()

            if score > bestScore:
              bestScore = score
              bestLen = len

          var seg: string
          for j in 0..<bestLen:
            seg.add($cjkRunes[i + j])
          result.add(seg)
          i += bestLen

proc tokenizeLine*(self: FastText; line: string): TokenizedResult =
  let tokens = self.segmentText(line)
  
  result.tokens = @[]
  result.unknownTokens = @[]
  
  for token in tokens:
    if token == EOS or token.len == 0:
      continue
      
    var tok = Token()
    tok.text = token
    tok.id = self.dict[].getId(token)
    
    var substrings: seq[string]
    var ngrams: seq[int32]
    self.dict[].getSubwords(token, ngrams, substrings)
    
    tok.subwords = substrings
    tok.subwordIds = ngrams
    
    if tok.id < 0 and substrings.len == 0:
      result.unknownTokens.add(token)
    
    result.tokens.add(tok)

proc tokenizeLines*(self: FastText; lines: seq[string]): seq[TokenizedResult] =
  for line in lines:
    result.add(self.tokenizeLine(line))
