-- Transducer:
require 'tardis.LSTM'

local Transducer, parent = torch.class('nn.Transducer', 'nn.Module')

function Transducer:__init(vocabSize, inputSize, hiddenSize, numLayers, dropout, rememberStates)
    -- alias
    local V, D, H = vocabSize, inputSize, hiddenSize
    self.dropout = dropout
    self.rememberStates = rememberStates

    self._rnns = {}
    self.net = nn.Sequential()
    self.net:add(nn.LookupTable(V, D))
    for i = 1, numLayers do
        local prevSize = H
        if i == 1 then prevSize = D end
        local rnn = nn.LSTM(prevSize, H)
        if self.rememberStates then
            rnn.rememberStates = true
        end
        table.insert(self._rnns, rnn)
        self.net:add(rnn)
        if self.dropout then
            self.net:add(nn.Dropout(self.dropout))
        end
    end
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
    local states = {}
    for _, rnn in ipairs(self._rnns) do
        table.insert(states, rnn:lastStates())
    end
    return states
end

function Transducer:setStates(states)
    for i, s in ipairs(states) do
        self._rnns[i]:setStates(s)
    end
end

function Transducer:gradStates()
    local gradStates = {}
    for _, rnn in ipairs(self._rnns) do
        table.insert(gradStates, rnn:gradStates())
    end
    return gradStates
end

function Transducer:setGradStates(gradStates)
    for i, grad in ipairs(gradStates) do
        self._rnns[i]:setGradStates(grad)
    end
end

function Transducer:indexStates(index)
    for _, rnn in ipairs(self._rnns) do
        rnn.rememberStates = true
        rnn:indexStates(index)
    end
end

function Transducer:updateGradInput(input, gradOutput)
    self:backward(input, gradOutput, 0)
end

function Transducer:accGradParameters(input, gradOutput, scale)
    self:backward(input, gradOutput, scale)
end

function Transducer:clearState()
    for _, rnn in ipairs(self._rnns) do
        rnn:clearState()
    end
end
