require 'cudnn'
local factory = require 'misc.factory'

local CharTransducer, parent = torch.class('nn.CharTransducer', 'nn.Module')

function CharTransducer:__init(opt)
    self.word2char = opt.word2char:cuda() -- word2char
    local nchars = self.word2char:max()
    local maxlen = self.word2char:size(2)
    local inputSize = opt.inputSize

    self._rnn = cudnn.LSTM(opt.hiddenSize, opt.hiddenSize, opt.numLayers, true, opt.dropout)
    self.net = nn.Sequential()
    self.net:add(nn.LookupTable(nchars, opt.inputSize))
    self.net:add(nn.Sum(2))
    self.net:add(nn.Tanh())
    --self.net:add(nn.Linear(inputSize, opt.hiddenSize))
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

function CharTransducer:parameters()
    return self.net:parameters()
end

function CharTransducer:training()
    self.net:training()
    parent.training(self)
end

function CharTransducer:evaluate()
    self.net:evaluate()
    parent.evaluate(self)
end

function CharTransducer:lastStates()
    local c = self._rnn.cellOutput
    local h = self._rnn.hiddenOutput
    return {c, h}
end

function CharTransducer:setStates(states)
    local c, h = unpack(states)
    if not self._rnn.cellInput then
        self._rnn.cellInput = c.new()
        self._rnn.hiddenInput = h.new()
    end
    self._rnn.cellInput:resizeAs(c):copy(c)
    self._rnn.hiddenInput:resizeAs(h):copy(h)
end

function CharTransducer:gradStates()
    local grad_c = self._rnn.gradCellInput
    local grad_h = self._rnn.gradHiddenInput
    return {grad_c, grad_h}
end

function CharTransducer:setGradStates(gradStates)
    local grad_c, grad_h = unpack(gradStates)
    if not self._rnn.gradCellOutput then
        self._rnn.gradCellOutput = grad_c.new()
        self._rnn.gradHiddenOutput = grad_h.new()
    end
    self._rnn.gradCellOutput:resizeAs(grad_c):copy(grad_c)
    self._rnn.gradHiddenOutput:resizeAs(grad_h):copy(grad_h)
end

function CharTransducer:indexStates(idx)
    self._rnn.rememberStates = true

    self._rnn.cellOutput    = self._rnn.cellOutput:index(2, idx)
    self._rnn.hiddenOutput  = self._rnn.hiddenOutput:index(2, idx)
    self._rnn.cellInput     = self._rnn.cellInput:index(2, idx)
    self._rnn.hiddenInput   = self._rnn.hiddenInput:index(2, idx)
end

function CharTransducer:updateGradInput(input, gradOutput)
    self:backward(input, gradOutput, 0)
end

function CharTransducer:accGradParameters(input, gradOutput, scale)
    self:backward(input, gradOutput, scale)
end

function CharTransducer:clearState()
    self._rnn:clearState()
end
