
import os
import unittest
import fasttext

const testDir = currentSourcePath().parentDir / "testdata"

suite "FastText Language Identification Tests (FTZ model)":
  var ft: FastText
  
  setup:
    ft = newFastText()
    ft.loadModel(testDir / "lid.176.ftz")
  
  test "model parameters are correct":
    check ft.args.lrUpdateRate == 100
    check ft.args.dim == 16
    check ft.args.minn == 2
    check ft.args.maxn == 4
    check ft.args.bucket == 2000000
  
  test "predict Malayalam language":
    let ss = "അമ്മ"
    let output = ft.predict(ss)
    check output.len > 0
    check output[0].second == "__label__ml"
    check output[0].first == 0.9535570740699768
