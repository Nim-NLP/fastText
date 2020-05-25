# download bin model from https://fasttext.cc/docs/en/language-identification.html

import os
import fasttext
import math

var ft = initFastText()
if existsFile("tests" / "lid.176.bin"):
    ft.loadModel("tests" / "lid.176.bin")

    var 
        print_prob = false
        k:int32 = 5
        threshold:float32 = 0.0
        # output:seq[tuple[first:float32, second:string]]
        ss = "അമ്മ"

    # ft.predict(ss,k,print_prob,threshold)

    assert ft.args.lrUpdateRate == 100
    assert ft.args.dim == 16
    assert ft.args.minn == 2
    assert ft.args.maxn == 4
    assert ft.args.bucket == 2000000

    # -0.01967973634600639
    let output = ft.predict(ss)
    assert output[0].second == "__label__ml"
    assert output[0].first == 0.9998614192008972