Porting [fastText](https://github.com/facebookresearch/fastText) in Nim.


**Note:This implementation supports prediction for supervised and unsupervised models, whether they are quantized or not. Please use C++ version of fastText for train, test and quantization.**

## Tokenizer API

This Nim port includes a **built-in tokenizer** that uses FastText's learned word embeddings to segment text.

**Note:** The quality of segmentation depends entirely on the vocabulary of the loaded FastText model. This is not a general-purpose tokenizer like jieba or spaCy - it simply looks up words in the model's dictionary.

### Features

- Dictionary-based matching using FastText's vocabulary
- Greedy longest-match algorithm for CJK text
- Vector norm fallback for unknown sequences
- Returns token IDs and subword n-gram information

### Usage

```nim
import fasttext

var ft = newFastText()
ft.loadModel("path/to/model.ftz")

# Tokenize text
let tokens = ft.tokenizeLine("Hello world! 这是一个测试。")
for token in tokens:
  echo token.text, " (id: ", token.id, ")"
```

The tokenizer categorizes text by character type (CJK, Latin, digits, punctuation), then performs dictionary lookup for CJK sequences. For unknown words, it uses vector norm scoring to determine boundaries.  

__Installation__

fastText can be installed via [Nimble](https://github.com/nim-lang/nimble):

```

> nimble install https://github.com/bung87/fastText
```

__Credits__

all licensing terms of [fastText](https://github.com/facebookresearch/fastText/blob/master/LICENSE) apply to the usage of this package.

## References

Please cite [1](#enriching-word-vectors-with-subword-information) if using this code for learning word representations or [2](#bag-of-tricks-for-efficient-text-classification) if using for text classification.

### Enriching Word Vectors with Subword Information

[1] P. Bojanowski\*, E. Grave\*, A. Joulin, T. Mikolov, [*Enriching Word Vectors with Subword Information*](https://arxiv.org/abs/1607.04606)

```
@article{bojanowski2017enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={Transactions of the Association for Computational Linguistics},
  volume={5},
  year={2017},
  issn={2307-387X},
  pages={135--146}
}
```

### Bag of Tricks for Efficient Text Classification

[2] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/abs/1607.01759)

```
@InProceedings{joulin2017bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  booktitle={Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers},
  month={April},
  year={2017},
  publisher={Association for Computational Linguistics},
  pages={427--431},
}
```

### FastText.zip: Compressing text classification models

[3] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, [*FastText.zip: Compressing text classification models*](https://arxiv.org/abs/1612.03651)

```
@article{joulin2016fasttext,
  title={FastText.zip: Compressing text classification models},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1612.03651},
  year={2016}
}
```

(\* These authors contributed equally.)


## Join the fastText community

* Facebook page: https://www.facebook.com/groups/1174547215919768
* Google group: https://groups.google.com/forum/#!forum/fasttext-library
* Contact: [egrave@fb.com](mailto:egrave@fb.com), [bojanowski@fb.com](mailto:bojanowski@fb.com), [ajoulin@fb.com](mailto:ajoulin@fb.com), [tmikolov@fb.com](mailto:tmikolov@fb.com)

See the CONTRIBUTING file for information about how to help out.

## License

fastText is BSD-licensed. We also provide an additional patent grant.