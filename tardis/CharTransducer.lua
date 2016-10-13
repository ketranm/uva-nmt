require 'tardis.FastTransducer'
local factory = require 'misc.factory'

local CharTransducer, parent = torch.class('nn.CharTransducer', 'nn.Transducer')

function CharTransducer:__init(opt)
    self.word2char = opt.word2char
    local nchars = self.word2char:max()
    local maxlen = self.word2char:size(2)

    local char_cnn = factory.build_cnn(opt.featureMaps, opt.kernels,
                                        opt.charSize, opt.hiddenSize, nchars, maxlen)

    local inputSize = torch.Tensor(opt.featureMaps):sum()
    self._rnn = cudnn.LSTM(inputSize, opt.hiddenSize, opt.numLayers, true, opt.dropout)
    self.net = nn.Sequential()
    self.net:add(char_cnn)
    self.view = nn.View()
    self.net:add(self.view)
    self.net:add(nn.Contiguous())
    self.net:add(self._rnn)
    self._input = torch.Tensor()
    self.hiddenSize = opt.hiddenSize
end

function CharTransducer:updateOutput(input)
    local N, T = input:size(1), input:size(2)
    self.view:resetSize(N, T, -1)
    self._input:index(self.word2char, 1, input:view(-1))

    return self.net:forward(self._input)
end

function CharTransducer:backward(input, gradOutput, scale)
    return self.net:backward(self._input, gradOutput, scale)
end
