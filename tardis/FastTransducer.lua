require 'cudnn'

local Transducer, parent = torch.class('nn.Transducer', 'nn.Module')

function Transducer:__init(vocabSize, inputSize, hiddenSize, numLayers, dropout, rememberStates)
    self._rnn = cudnn.LSTM(inputSize, hiddenSize, numLayers, true, dropout)
    self.net = nn.Sequential()
    self.net:add(nn.LookupTable(vocabSize, inputSize))
    self.net:add(self._rnn)
end

function Transducer:updateOutput(input)
    return self.net:forward(input)
end

function Transducer:backward(input, gradOutput, scale)
    return self.net:backward(input, gradOutput, scale)
end

function Transducer:parameters()
    return self.net:parameters()
end

function Transducer:training()
    self.net:training()
    parent.training(self)
end

function Transducer:evaluate()
    self.net:evaluate()
    parent.evaluate(self)
end

function Transducer:lastStates()
    local c = self._rnn.cellOutput
    local h = self._rnn.hiddenOutput
    return {c, h}
end

function Transducer:setStates(states)
    local c, h = unpack(states)
    if not self._rnn.cellInput then
        self._rnn.cellInput = c.new()
        self._rnn.hiddenInput = h.new()
    end
    self._rnn.cellInput:resizeAs(c):copy(c)
    self._rnn.hiddenInput:resizeAs(h):copy(h)
end

function Transducer:gradStates()
    local grad_c = self._rnn.gradCellInput
    local grad_h = self._rnn.gradHiddenInput
    return {grad_c, grad_h}
end

function Transducer:setGradStates(gradStates)
    local grad_c, grad_h = unpack(gradStates)
    self._rnn.gradCellInput:resizeAs(grad_c):copy(grad_c)
    self._rnn.gradHiddenInput:resizeAs(grad_h):copy(grad_h)
end

function Transducer:indexStates(index)
    self._rnn.rememberStates = true
    local c_out = self._rnn.cellOutput
    local h_out = self._rnn.hiddenOutput
    local c_inp = self._rnn.cellInput
    local h_inp = self._rnn.hiddenInput

    c_out = c_out:index(2, index)
    h_out = h_out:index(2, index)
    c_inp = c_inp:index(2, index)
    h_inp = h_inp:index(2, index)
end

function Transducer:updateGradInput(input, gradOutput)
    self:backward(input, gradOutput, 0)
end

function Transducer:accGradParameters(input, gradOutput, scale)
    self:backward(input, gradOutput, scale)
end

function Transducer:clearState()
    self._rnn:clearState()
end
