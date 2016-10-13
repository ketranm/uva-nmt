# UvA-NMT
University of Amsterdam's Neural Machine Translation

# User Guide

Tardis is mainly developed for NLP, therefore its interfaces have NLPish feel.

## Installation
Tardis is written for GPU, so you need to use GPUs in order to run Tardis. Additionally you need the following packages

* [Torch](http://torch.ch/)
* [moses](https://github.com/Yonaba/Moses)
* [lua-utf8](https://github.com/starwing/luautf8)
* cudnn

Go to NVIDIA homepage and download CUDNN version that corresponds to your CUDA (7.5 if you are using DAS5)

That's all!

## Code Organization

### Data Processing
Most of the code for processing data is in `data/`

* [AbstractDataLoader.lua](https://github.com/ketranm/uva-nmt/blob/master/data/AbstractDataLoader.lua) provides general text processing utilities such as creating vocabulary, load and shuffle data, convert text to tensor and vice-visa,...
* [loadBitext.lua](https://github.com/ketranm/uva-nmt/blob/master/data/loadBitext.lua) inherits from  [AbstractDataLoader.lua](https://github.com/ketranm/uva-nmt/blob/master/data/AbstractDataLoader.lua). It provides general API for bitext processing
* [loadText.lua](https://github.com/ketranm/uva-nmt/blob/master/data/loadText.lua) inherits from AbstractDataLoader. It is useful for Language Model.

### TARDIS
`tardis/` contains all most everything for Seq2seq models.

* [Transducer](https://github.com/ketranm/uva-nmt/blob/master/tardis/Transducer.lua), consumes a sequence of input and produce a sequence of hidden states. Transducer can be used for language model and it is the basic block to build encoder/decoder in Seq2seq.
* [FastTransducer](https://github.com/ketranm/uva-nmt/blob/master/tardis/FastTransducer.lua), which is fast. It uses `cudnn`. It has the same set of APIs as Transducer.
* [SeqAtt.lua](https://github.com/ketranm/uva-nmt/blob/master/tardis/SeqAtt.lua) or NMT, which is a standard seq2seq architecture with [attention](https://github.com/ketranm/uva-nmt/blob/master/tardis/GlimpseDot.lua).

## Using Tardis
First, you need a configuration file to describe your experimental setup.
see `iwslt.de2en.conf` for an example of config file.

Once you have your config file `expr.conf` you can run tardis
```
$ th main.lua expr.conf
```

## Features

* Seq2seq with attention
* Fast LSTM / enable `cudnn`
* Hard Attention
* REINFORCE
* [Conditional Attention GRU](https://github.com/nyu-dl/dl4mt-tutorial/blob/master/docs/cgru.pdf)

## Contributors
* Ke Tran
* Arianna Bisazza
