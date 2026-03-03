import os
import unittest
import fasttext

const testDir = currentSourcePath().parentDir / "testdata"

suite "FastText News Classification Tests":
  var ft: FastText
  
  setup:
    ft = newFastText()
    ft.loadModel(testDir / "sogou_news.ftz")
  
  test "model parameters are correct":
    check ft.args.lrUpdateRate == 100
    check ft.args.dim == 10
    check ft.args.minn == 0
    check ft.args.maxn == 0
    check ft.args.bucket == 10000000
  
  test "classify sports news (pinyin input)":
    let ss = "be3i ji1ng shi2 jia1n 5 yue4 16 ri4 ， 2018 nia2n NBA xua3n xiu4 le4 to4u cho1u qia1n da4 hui4 za4i zhi1 jia1 ge1 ji4n xi2ng 。 zui4 zho1ng ， fe1i ni2 ke4 si1 ta4i ya2ng xi4ng yu4n cho1u da4o zhua4ng yua2n qia1n ， e2r ba3ng ya3n he2 ta4n hua1 qia1n ze2 fe1n bie2 gui1 shu3 sa4 ke4 la1 me2n tuo1 guo2 wa2ng he2 ya4 te4 la2n da4 la3o yi1ng"
    let output = ft.predict(ss)
    check output.len > 0
    check output[0].second == "__label__1"  # sports label
  
  test "classify technology news (pinyin input)":
    let s2 = "me3i guo2 ha2ng tia1n ju2 ji4n ri4 za4i qi2 gua1n wa3ng bo2 ke4 sha4ng go1ng bu4 le sha1ng ye4 za4i re2n fe1i chua2n de shi4 ya4n xi4ng fe1i xi2ng shi2 jia1n bia3o ， qi2 zho1ng di4 yi1 ci4 shi4 fe1i jia1ng bu4 za4i re2n ， yo2u me3i guo2 ta4i ko1ng ta4n suo3 ji4 shu4 go1ng si1 de lo2ng fe1i chua2n yu2 mi2ng nia2n nia2n chu1 zhi2 xi2ng 。"
    let output2 = ft.predict(s2)
    check output2.len > 0
    check output2[0].second == "__label__5"  # technology label
