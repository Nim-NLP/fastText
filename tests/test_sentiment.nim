import os
import unittest
import fasttext

const testDir = currentSourcePath().parentDir / "testdata"
const modelPath = testDir / "amazon_review_polarity.ftz"
const modelExists = fileExists(modelPath)

suite "FastText Sentiment Analysis Tests":
  var ft: FastText
  
  setup:
    if modelExists:
      ft = newFastText()
      ft.loadModel(modelPath)
  
  teardown:
    ft = nil
  
  test "model parameters are correct":
    if not modelExists:
      skip()
    else:
      check ft.args.lrUpdateRate == 100
      check ft.args.dim == 10
      check ft.args.minn == 0
      check ft.args.maxn == 0
      check ft.args.bucket == 10000000
  
  test "predict positive sentiment":
    if not modelExists:
      skip()
    else:
      let text = "This product is amazing! I love it and would definitely recommend to everyone."
      let output = ft.predict(text)
      check output.len > 0
      check output[0].second == "__label__2"  # positive label
  
  test "predict negative sentiment":
    if not modelExists:
      skip()
    else:
      let text = "Terrible product. Complete waste of money. Do not buy!"
      let output = ft.predict(text)
      check output.len > 0
      check output[0].second == "__label__1"  # negative label
