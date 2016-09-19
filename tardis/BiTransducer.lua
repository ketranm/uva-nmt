-- Bidirectional Transducer
require 'cudnn'

local BiTransducer, parent = torch.class('nn.BiTransducer', 'nn.Module')

function BiTransducer:__init(vocabSize, inputSize, hiddenSize, numLayers, dropout)
    self._rnn = cudnn.BLSTM(inputSize, hiddenSize, numLayers, true, dropout)
    self.view = nn.View()
    self.net = nn.Sequential()
    self.net:add(nn.LookupTable(vocabSize, inputSize))
    self.net:add(self._rnn)
    self.net:add(nn.Contiguous())
    self.net:add(nn.View(-1, 2 * hiddenSize))
    self.net:add(nn.Linear(2 * hiddenSize, hiddenSize, false))  -- nobias
    self.net:add(self.view)
    self.keep = 'forward'
end

function BiTransducer:updateOutput(input)
    local N, T = input:size(1), input:size(2)
    self.view:resetSize(N, T, -1)
    return self.net:forward(input)
end

function BiTransducer:backward(input, gradOutput, scale)
    return self.net:backward(input, gradOutput, scale)
end

function BiTransducer:parameters()
    return self.net:parameters()
end

function BiTransducer:training()
    self.net:training()
    parent.training(self)
end

function BiTransducer:evaluate()
    self.net:evaluate()
    parent.evaluate(self)
end

function BiTransducer:lastStates()
    local c = self._rnn.cellOutput
    local h = self._rnn.hiddenOutput
    if self.keep == 'forward' then
        return {c[{{1}, {}, {}}], h[{{1}, {}, {}}]}
    elseif self.keep == 'backward' then
        return {c[{{2}, {}, {}}], h[{{2}, {}, {}}]}
    elseif self.keep == 'both' then
        return {c, h}
    else
        err('self.keep is incorrect!')
    end
end

function BiTransducer:setStates(states)
    local c, h = unpack(states)
    if not self._rnn.cellInput then
        self._rnn.cellInput = c.new()
        self._rnn.hiddenInput = h.new()
    end
    self._rnn.cellInput:resizeAs(c):copy(c)
    self._rnn.hiddenInput:resizeAs(h):copy(h)
end

function BiTransducer:gradStates()
    local grad_c = self._rnn.gradCellInput
    local grad_h = self._rnn.gradHiddenInput
    return {grad_c, grad_h}
end

function BiTransducer:setGradStates(gradStates)
    local grad_c, grad_h = unpack(gradStates)
    if not self._rnn.gradCellOutput then
        self._rnn.gradCellOutput = grad_c.new()
        self._rnn.gradHiddenOutput = grad_h.new()
    end
    local  L, H = grad_c:size(2), grad_c:size(3)
    assert(grad_h:size(2) == L)
    assert(grad_h:size(3) == H)
    if grad_c:size(1) == 2 then
        assert(grad_h:size(1) == 2)
        assert(self.keep == 'both')
        self._rnn.gradCellOutput:resizeAs(grad_c):copy(grad_c)
        self._rnn.gradHiddenOutput:resizeAs(grad_h):copy(grad_h)
    else
        self._rnn.gradCellOutput:resize(2, L, H):zero()
        self._rnn.gradHiddenOutput:resize(2, L, H):zero()
        local dim = 1
        if self.keep == 'backward' then dim = 2 end
        self._rnn.gradCellOutput[dim]:copy(grad_c)
        self._rnn.gradHiddenOutput[dim]:copy(grad_h)
end

function BiTransducer:indexStates(idx)
    self._rnn.rememberStates = true

    self._rnn.cellOutput    = self._rnn.cellOutput:index(2, idx)
    self._rnn.hiddenOutput  = self._rnn.hiddenOutput:index(2, idx)
    self._rnn.cellInput     = self._rnn.cellInput:index(2, idx)
    self._rnn.hiddenInput   = self._rnn.hiddenInput:index(2, idx)
end

function BiTransducer:updateGradInput(input, gradOutput)
    self:backward(input, gradOutput, 0)
end

function BiTransducer:accGradParameters(input, gradOutput, scale)
    self:backward(input, gradOutput, scale)
end

function BiTransducer:clearState()
    self._rnn:clearState()
end
