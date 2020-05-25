import os
import fasttext
import math

var ft = initFastText()

ft.loadModel("tests" / "sogou_news.ftz")

var 
    print_prob = false
    k:int32 = 5
    threshold:float32 = 0.0
    # ss = "北京时间5月16日，2018年NBA选秀乐透抽签大会在芝加哥进行。最终，菲尼克斯太阳幸运抽到状元签，而榜眼和探花签则分别归属萨克拉门托国王和亚特兰大老鹰"
    ss = "be3i ji1ng shi2 jia1n 5 yue4 16 ri4 ， 2018 nia2n NBA xua3n xiu4 le4 to4u cho1u qia1n da4 hui4 za4i zhi1 jia1 ge1 ji4n xi2ng 。 zui4 zho1ng ， fe1i ni2 ke4 si1 ta4i ya2ng xi4ng yu4n cho1u da4o zhua4ng yua2n qia1n ， e2r ba3ng ya3n he2 ta4n hua1 qia1n ze2 fe1n bie2 gui1 shu3 sa4 ke4 la1 me2n tuo1 guo2 wa2ng he2 ya4 te4 la2n da4 la3o yi1ng"

assert ft.args.lrUpdateRate == 100
assert ft.args.dim == 10
assert ft.args.minn == 0
assert ft.args.maxn == 0
assert ft.args.bucket == 10000000

# __label__1 sports
let output = ft.predict(ss)
assert output[0].second == "__label__1" 

# __label__5 technology
# var s2 = "美国航天局近日在其官网博客上公布了商业载人飞船的试验性飞行时间表，其中第一次试飞将不载人，由美国太空探索技术公司的“龙”飞船于明年年初执行。"
var s2 = "me3i guo2 ha2ng tia1n ju2 ji4n ri4 za4i qi2 gua1n wa3ng bo2 ke4 sha4ng go1ng bu4 le sha1ng ye4 za4i re2n fe1i chua2n de shi4 ya4n xi4ng fe1i xi2ng shi2 jia1n bia3o ， qi2 zho1ng di4 yi1 ci4 shi4 fe1i jia1ng bu4 za4i re2n ， yo2u me3i guo2 ta4i ko1ng ta4n suo3 ji4 shu4 go1ng si1 de “ lo2ng ” fe1i chua2n yu2 mi2ng nia2n nia2n chu1 zhi2 xi2ng 。"
let output2 = ft.predict(s2)
assert output2[0].second == "__label__5" 