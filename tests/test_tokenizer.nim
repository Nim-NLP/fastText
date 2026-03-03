import os
import fasttext
import tokenizer
import sequtils
import unittest

const testDir = currentSourcePath().parentDir / "testdata"
const modelPath = testDir / "chinese_bookmark_classifier.ftz"
const modelExists = fileExists(modelPath)

suite "FastText Tokenizer Tests":
  var ft: FastText

  const s1 = "北京时间5月16日，2018年NBA选秀乐透抽签大会在芝加哥进行。最终，菲尼克斯太阳幸运抽到状元签，而榜眼和探花签则分别归属萨克拉门托国王和亚特兰大老鹰"
  const s2 = "美国航天局近日在其官网博客上公布了商业载人飞船的试验性飞行时间表，其中第一次试飞将不载人，由美国太空探索技术公司的龙飞船于明年年初执行。"

  setup:
    if modelExists:
      ft = newFastText()
      ft.loadModel(modelPath)

  teardown:
    ft = nil

  test "tokenize s1 should produce exact expected tokens":
    if not modelExists:
      skip()
    else:
      let result = ft.tokenizeLine(s1)
      let tokens = result.mapIt(it.text)
      check tokens == @[
        "北京", "时间", "5", "月", "16", "日", "，", "2018", "年", "NBA",
        "选秀", "乐透", "抽签", "大会", "在", "芝加哥", "进行", "。",
        "最终", "，", "菲尼克斯", "太阳", "幸运", "抽到", "状元", "签",
        "，", "而", "榜眼", "和", "探花", "签", "则", "分别", "归属",
        "萨克拉门托", "国王", "和", "亚特兰大", "老鹰"
      ]

  test "tokenize s2 should produce exact expected tokens":
    if not modelExists:
      skip()
    else:
      let result = ft.tokenizeLine(s2)
      let tokens = result.mapIt(it.text)
      check tokens == @[
        "美国", "航天", "局", "近日", "在", "其", "官网", "博客", "上",
        "公布", "了", "商业", "载人", "飞船", "的", "试验性", "飞行",
        "时间表", "，", "其中", "第一次", "试飞", "将", "不载人", "，",
        "由", "美国", "太空", "探索", "技术", "公司", "的", "龙", "飞船",
        "于", "明年", "年初", "执行", "。"
      ]
