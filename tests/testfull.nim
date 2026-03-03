# download bin model from https://fasttext.cc/docs/en/language-identification.html

import os
import unittest
import fasttext

const testDir = currentSourcePath().parentDir / "testdata"
const binModelExists = fileExists(testDir / "lid.176.bin")

suite "FastText Language Identification Tests (BIN model)":
  var ft: FastText
  
  setup:
    ft = newFastText()
    if binModelExists:
      ft.loadModel(testDir / "lid.176.bin")
  
  test "model file exists and loads correctly":
    if not binModelExists:
      skip()
    check binModelExists == true
  
  test "model parameters are correct":
    if not binModelExists:
      skip()
    check ft.args.lrUpdateRate == 100
    check ft.args.dim == 16
    check ft.args.minn == 2
    check ft.args.maxn == 4
    check ft.args.bucket == 2000000
  
  test "predict Malayalam language":
    if not binModelExists:
      skip()
    let ss = "അമ്മ"
    let output = ft.predict(ss)
    check output.len > 0
    check output[0].second == "__label__ml"
    check output[0].first == 0.9998614192008972
