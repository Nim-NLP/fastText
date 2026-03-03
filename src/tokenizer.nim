## Tokenizer module for FastText word segmentation.
##
## This module provides a built-in tokenizer that uses FastText's learned
## word embeddings to segment text. It leverages the model's internal
## vocabulary and subword vectors for segmentation.
##
## Primary API:
## - `tokenizeLine`: Tokenizes a single line of text into a sequence of tokens
##   with their IDs and subword information.
##
## Limitations:
## - The quality of segmentation depends entirely on the vocabulary of the
##   loaded FastText model. If the model was trained with suboptimal
##   segmentation, this tokenizer will produce similar results.
## - For CJK text, the algorithm uses a greedy longest-match strategy which
##   may not always produce linguistically optimal segmentation.
##
## Example usage:
##   ```nim
##   var ft = initFastText()
##   ft.loadModel("model.bin")
##   let tokens = ft.tokenizeLine("Hello world")
##   for tok in tokens:
##     echo tok.text, " ", tok.id
##   ```

import unicode
import streams
import ./dictionary
import ./types
import ./vector


type
  Token* = object
    ## Represents a single token with its metadata.
    text*: string         ## The token string.
    id*: int32            ## Vocabulary ID (-1 if unknown).
    subwordIds*: seq[int32]  ## Subword n-gram IDs for this token.


proc isCjkRune(r: Rune): bool {.inline.} =
  ## Checks if a rune is a CJK (Chinese, Japanese, Korean) unified ideograph.
  let code = r.int32
  result = (code >= 0x4E00 and code <= 0x9FFF) or
       (code >= 0x3400 and code <= 0x4DBF) or
       (code >= 0x20000 and code <= 0x2A6DF)

proc isPunctuationRune(r: Rune): bool {.inline.} =
  ## Checks if a rune is a punctuation mark (ASCII or CJK).
  if r.int32 <= 127:
    result = chr(r.int32) in {',', '.', '!', '?', ';', ':', '"', '\'', '(', ')',
      '[', ']', '{', '}', '<', '>'}
  else:
    result = r.int32 in [
      0x3001, 0x3002, 0xFF0C, 0xFF0E, 0xFF01, 0xFF1F, 0xFF1B, 0xFF1A,
      0x201C, 0x201D, 0xFF08, 0xFF09, 0x3010, 0x3011, 0x300A, 0x300B,
      0x2014, 0x2026
    ]

proc isDigitRune(r: Rune): bool {.inline.} =
  ## Checks if a rune is an ASCII digit (0-9).
  result = r.int32 >= ord('0') and r.int32 <= ord('9')

type
  TokenInfo = object
    ## Internal temporary structure for token accumulation.
    isCjk: bool
    runes: seq[Rune]

proc getWordVector(self: FastText; word: string): Vector =
  ## Computes the word vector by averaging its subword n-gram vectors.
  result = newVector(self.args.dim.int64)
  let ngrams = self.dict.getSubwords(word)
  var count = 0
  if ngrams.len > 0:
    for ngram in ngrams:
      if ngram >= 0 and ngram < self.input.m:
        result.addRow(self.input, ngram.int64)
        inc count
    if count > 0:
      result.mul(1.0 / count.float32)

proc segmentText*(self: FastText; text: string): seq[string] =
  ## Segments text into words using dictionary-based matching.
  ##
  ## For CJK characters, uses a greedy longest-match algorithm with
  ## vector-based fallback for unknown character sequences.
  if text.len == 0:
    return @[]

  let runes = text.toRunes

  var tokens: seq[TokenInfo]
  var i = 0
  while i < runes.len:
    let r = runes[i]

    # Note: Check order matters - CJK must be checked before isAlpha
    # because unicode.isAlpha returns true for CJK characters.
    if r.isCjkRune():
      var t = TokenInfo(isCjk: true, runes: newSeqOfCap[Rune](16))
      while i < runes.len and runes[i].isCjkRune():
        t.runes.add(runes[i])
        inc i
      tokens.add(t)
    elif r.isDigitRune():
      var t = TokenInfo(isCjk: false, runes: newSeqOfCap[Rune](8))
      while i < runes.len and runes[i].isDigitRune():
        t.runes.add(runes[i])
        inc i
      tokens.add(t)
    elif r.isAlpha():
      var t = TokenInfo(isCjk: false, runes: newSeqOfCap[Rune](16))
      while i < runes.len and runes[i].isAlpha():
        t.runes.add(runes[i])
        inc i
      tokens.add(t)
    elif r.isPunctuationRune():
      var t = TokenInfo(isCjk: false, runes: newSeq[Rune](1))
      t.runes[0] = r
      tokens.add(t)
      inc i
    else:
      if not r.isWhiteSpace():
        var t = TokenInfo(isCjk: false, runes: newSeq[Rune](1))
        t.runes[0] = r
        tokens.add(t)
      inc i

  for token in tokens:
    if not token.isCjk:
      result.add($token.runes)
    else:
      let cjkRunes = token.runes
      let n = cjkRunes.len
      if n == 0:
        continue

      # Convert once to string, then use runeSubStr for substring extraction
      let cjkStr = $cjkRunes

      var pos = 0
      while pos < n:
        var found = false
        let maxWordLen = min(6, n - pos)
        for len in countdown(maxWordLen, 1):
          let candidate = runeSubStr(cjkStr, pos, len)
          let wordId = self.dict.getId(candidate)
          if wordId >= 0:
            result.add(candidate)
            pos += len
            found = true
            break

        if not found:
          var bestLen = 1
          var bestScore = -1.0

          for len in 1..min(3, n - pos):
            let candidate = runeSubStr(cjkStr, pos, len)
            let vec = self.getWordVector(candidate)
            let score = vec.norm()
            if score > bestScore:
              bestScore = score
              bestLen = len

          result.add(runeSubStr(cjkStr, pos, bestLen))
          pos += bestLen

proc tokenizeLine*(self: FastText; line: string): seq[Token] =
  ## Tokenizes a single line of text into a sequence of tokens.
  ##
  ## Each token contains the text, vocabulary ID, and subword n-gram IDs.
  ## Unknown words (not in vocabulary) will have id = -1.
  let tokens = self.segmentText(line)
  result = newSeqOfCap[Token](tokens.len)

  for token in tokens:
    if token == EOS or token.len == 0:
      continue

    var tok = Token()
    tok.text = token
    tok.id = self.dict.getId(token)
    tok.subwordIds = self.dict.getSubwords(token)
    result.add(tok)

iterator tokenizeStream*(self: FastText; input: Stream): seq[Token] =
  ## Tokenizes a stream line by line.
  ##
  ## This is memory-efficient for large files as it processes one line at a time
  ## instead of loading everything into memory.
  if not input.isNil:
    var line: string
    while input.readLine(line):
      yield self.tokenizeLine(line)

proc tokenizeLines*(self: FastText; lines: openArray[string]): seq[seq[Token]] =
  ## Tokenizes multiple lines in batch.
  ##
  ## Uses `openArray` for flexibility - accepts arrays, sequences, or slices.
  result = newSeqOfCap[seq[Token]](lines.len)
  for line in lines:
    result.add(self.tokenizeLine(line))
