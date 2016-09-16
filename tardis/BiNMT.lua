-- using Bidirectional Encoder
-- author: Ke Tran <m.k.tran@uva.nl>

require 'GlimpseDot'
require 'optim'
require 'cudnn'
local model_utils = require 'core.model_utils'

local utils = require 'misc.utils'

local NMT, parent = torch.class('nn.NMT', 'nn.Module')

function NMT:__init(opt)
    -- build encoder
    local srcVocabSize = opt.srcVocabSize
    local embeddingSize = opt.embeddingSize
    local hiddenSize = opt.hiddenSize
    self.view1 = nn.View()
    self.view2 = nn.View()
    self.encoder = nn.Sequential()
    self.encoder:add(nn.LookupTable(srcVocabSize, embeddingSize))
    self.encoder:add(cudnn.BGRU(embeddingSize, hiddenSize, 1, true))
    self.encoder:add(nn.Contiguous())
    self.encoder:add(self.view1)
    self.encoder:add(nn.Linear(2 * hiddenSize, hiddenSize, false))
    self.encoder:add(self.view2)

    -- build decoder
    local trgVocabSize = opt.trgVocabSize
    self.decoder = nn.Sequential()
    self.decoder:add(nn.LookupTable(trgVocabSize, embeddingSize))
    self.decoder:add(cudnn.GRU(embeddingSize, hiddenSize, 1, true))

    self.glimpse = nn.GlimpseDot(opt.hiddenSize)

    self.layer = nn.Sequential()
    self.layer:add(nn.JoinTable(3))
    self.layer:add(nn.View(-1, 2 * hiddenSize))
    self.layer:add(nn.Linear(2 * hiddenSize, hiddenSize, false))
    self.layer:add(nn.Tanh(1, true))
    self.layer:add(nn.Linear(hiddenSize, trgVocabSize, true))
    self.layer:add(nn.LogSoftMax())

    local weights = torch.ones(trgVocabSize)
    weights[opt.pad_idx] = 0

    self.sizeAverage = true
    self.pad_idx = opt.pad_idx
    self.criterion = nn.ClassNLLCriterion(weights, false)
    self.tot = torch.CudaTensor() -- count non padding symbols
    self.numSamples = 0

    -- convert to cuda
    self.encoder:cuda()
    self.decoder:cuda()
    self.glimpse:cuda()
    self.layer:cuda()
    self.criterion:cuda()

    self.params, self.gradParams =
        model_utils.combine_all_parameters(self.encoder,
                                           self.decoder,
                                           self.glimpse,
                                           self.layer)
    self.maxNorm = opt.maxNorm or 5
    -- for optim
    self.optimConfig = {}
    self.optimStates = {}
    -- use buffer to store all the information needed for forward/backward
    self.buffers = {}
    self.output = torch.LongTensor()
end

function NMT:forward(input, target)
    --[[ Forward pass of NMT

    Parameters:
    - `input` : table of source and target tensor
    - `target` : a tensor of next words

    Return:
    - `logProb` : negative log-likelihood of the mini-batch
    --]]
    local target = target:view(-1)
    self:stepEncoder(input[1])

    local logProb = self:stepDecoder(input[2])
    self.tot:resizeAs(target)
    self.tot:ne(target, self.pad_idx)
    self.numSamples = self.tot:sum()
    local nll = self.criterion:forward(logProb, target)
    return nll / self.numSamples

end

function NMT:backward(input, target, gradOutput)
    -- zero grad manually here
    self.gradParams:zero()

    local buffers = self.buffers
    local outputEncoder = buffers.outputEncoder
    local outputDecoder = buffers.outputDecoder
    local context = buffers.context
    local logProb = buffers.logProb

    -- all good. Ready to back-prop
    local gradLoss = gradOutput
    -- by default, we use Cross-Entropy loss
    if not gradLoss then
        gradLoss = self.criterion:backward(logProb, target:view(-1))
        local norm_coeff = 1 / (self.sizeAverage and self.numSamples or 1)
        gradLoss:mul(norm_coeff)
    end

    local gradLayer = self.layer:backward({context, outputDecoder}, gradLoss)
    local gradDecoder = gradLayer[2] -- grad to decoder
    local gradGlimpse =
        self.glimpse:backward({outputEncoder, outputDecoder}, gradLayer[1])

    gradDecoder:add(gradGlimpse[2]) -- accumulate gradient in-place

    self.decoder:backward(input[2], gradDecoder)

    -- initialize gradient from decoder
    local gradHiddenInputDec = self.decoder:get(2).gradHiddenInput
    assert(gradHiddenInputDec:dim() == 3)
    local N = gradHiddenInputDec:size(2)
    local D = gradHiddenInputDec:size(3)

    local gradHiddenInputEnc = self.encoder:get(2).gradHiddenInput
    gradHiddenInputEnc:resize(2, N, D):zero()
    gradHiddenInputEnc[1]:copy(gradHiddenInputDec)
    -- backward to encoder
    local gradEncoder = gradGlimpse[1]
    self.encoder:backward(input[1], gradEncoder)
end

function NMT:update(learningRate)
    local gradNorm = self.gradParams:norm()
    local scale = learningRate
    if gradNorm > self.maxNorm then
        scale = scale * self.maxNorm / gradNorm
    end
    self.params:add(self.gradParams:mul(-scale)) -- do it in-place
end

function NMT:optimize(input, target)
    local feval = function(x)
        if self.params ~= x then
            self.params:copy(x)
        end
        local f = self:forward(input, target)
        self:backward(input, target)
        return f, self.gradParams
    end
    local _, fx = optim.adam(feval, self.params, self.optimConfig, self.optimStates)
    return fx[1]
end

function NMT:parameters()
    return self.params
end

function NMT:training()
    self.encoder:training()
    self.decoder:training()
end

function NMT:evaluate()
    self.encoder:evaluate()
    self.decoder:evaluate()
end

function NMT:load(fileName)
    local params = torch.load(fileName)
    self.params:copy(params)
end

function NMT:save(fileName)
    torch.save(fileName, self.params)
end

-- useful interface for beam search
function NMT:stepEncoder(x)
    --[[ Encode the source sequence
    All the information produced by the encoder is stored in buffers
    Parameters:
    - `x` : source tensor, can be a matrix (batch)
    --]]
    local N, T = x:size(1), x:size(2)
    self.view1:resetSize(N * T, -1)
    self.view2:resetSize(N, T, -1)
    local outputEncoder = self.encoder:forward(x)
    local prevState = self.encoder:get(2).hiddenOutput:select(1, 1)
    self.buffers = {outputEncoder = outputEncoder, prevState = prevState}
end

function NMT:stepDecoder(x)
    --[[ Run the decoder
    If it is called for the first time, the decoder will be initialized
    from the last state of the encoder. Otherwise, it will continue from
    its last state. This is useful for beam search or reinforce training

    Parameters:
    - `x` : target sequence, can be a matrix (batch)

    Return:
    - `logProb` : cross entropy loss of the sequence
    --]]

    -- get out necessary information from the buffers
    local buffers = self.buffers
    local outputEncoder, prevState = buffers.outputEncoder, buffers.prevState
    --
    local hidDec = self.decoder:get(2).hiddenOutput
    hidDec:resizeAs(prevState):copy(prevState)
    local outputDecoder = self.decoder:forward(x)
    local context = self.glimpse:forward({outputEncoder, outputDecoder})
    local logProb = self.layer:forward({context, outputDecoder})

    -- update buffer, adding information needed for backward pass
    buffers.outputDecoder = outputDecoder
    buffers.prevState = self.decoder:get(2).hiddenOutput
    buffers.context = context
    buffers.logProb = logProb

    return logProb
end

-- REINFORCE training Neural Machine Translation
function NMT:sample(nsteps, k)
    --[[ Sample `nsteps` from the model
    Assume that we already run the encoder and started reading in <s> symbol,
    so the buffers must contain log probability of the next words

    Parameters:
    - `nsteps` : integer, number of time steps
    - `k` : sample from top k
    Returns:
    - `output` : 2D tensor of sampled words
    --]]

    if not self.prob then
        self.prob = torch.Tensor():typeAs(self.params)
    end

    local buffers = self.buffers
    local outputEncoder, prevState = buffers.outputDecoder, buffers.prevState
    self.decoder:initState(prevState)

    local logProb = buffers.logProb  -- from the previous prediction
    assert(logProb ~= nil)
    local batchSize = outputEncoder:size(1)
    self.output:resize(batchSize, nsteps)

    for i = 1, nsteps do
        self.prob:resizeAs(logProb)
        self.prob:copy(logProb)
        self.prob:exp()
        if k then
            local prob_k, idx = self.prob:topk(k, true)
            prob_k:cdiv(prob_k:sum(2):repeatTensor(1, k)) -- renormalized
            local sample = torch.multinomial(prob_k, 1)
            self.output[{{}, {i}}] = idx:gather(2, sample)
        else
            self.prob.multinomial(self.output[{{}, {i}}], self.prob, 1)
        end
        logProb = self:stepDecoder(self.output[{{},{i}}])
    end

    return self.output
end

function NMT:indexDecoderState(index)
    --[[ This method is useful for beam search.
    It is similar to torch.index function, return a new state of kept index

    Parameters:
    - `index` : torch.LongTensor object

    Return:
    - `state` : new hidden state of the decoder, indexed by the argument
    --]]

    local currState = self.decoder:lastState()
    local newState = {}
    for _, state in ipairs(currState) do
        local sk = {}
        for _, s in ipairs(state) do
            table.insert(sk, s:index(1, index))
        end
        table.insert(newState, sk)
    end

    -- here, it make sense to update the buffer as well
    local buffers = self.buffers
    buffers.prevState = newState
    buffers.outputEncoder = buffers.outputEncoder:index(1, index)

    return newState
end

function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.layer:clearState()
end
