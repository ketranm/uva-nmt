-- using Bidirectional Encoder
-- author: Ke Tran <m.k.tran@uva.nl>

require 'tardis.GlimpseDot'
require 'optim'
require 'cudnn'
local model_utils = require 'tardis.model_utils'

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
    self.encoder:add(cudnn.LSTM(embeddingSize, hiddenSize, opt.numLayers, true, opt.dropout))

    -- build decoder
    local trgVocabSize = opt.trgVocabSize
    self.decoder = nn.Sequential()
    self.decoder:add(nn.LookupTable(trgVocabSize, embeddingSize))
    self.decoder:add(cudnn.LSTM(embeddingSize, hiddenSize, opt.numLayers, true, opt.dropout))

    self.glimpse = nn.GlimpseDot(hiddenSize)

    self.layer = nn.Sequential()
    self.layer:add(nn.JoinTable(3))
    self.layer:add(nn.View(-1, 2 * hiddenSize))
    self.layer:add(nn.Linear(2 * hiddenSize, hiddenSize, false))
    self.layer:add(nn.Tanh())
    self.layer:add(nn.Linear(hiddenSize, trgVocabSize, true))
    self.layer:add(nn.LogSoftMax())

    local weights = torch.ones(trgVocabSize)
    weights[opt.padIdx] = 0

    self.sizeAverage = true
    self.padIdx = opt.padIdx
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
    self.tot:ne(target, self.padIdx)
    self.numSamples = self.tot:sum()
    local nll = self.criterion:forward(logProb, target)
    return nll / self.numSamples

end

function NMT:backward(input, target)
    -- zero grad manually here
    self.gradParams:zero()

    local buffers = self.buffers
    local outputEncoder = buffers.outputEncoder
    local outputDecoder = buffers.outputDecoder
    local context = buffers.context
    local logProb = buffers.logProb



    local gradLoss = self.criterion:backward(self.logProb, target:view(-1))

    local gradLayer = self.layer:backward({self.glimpseOutput, self.outputDecoder}, gradLoss)
    local gradDecoder = gradLayer[2] -- grad to decoder
    local gradGlimpse =
        self.glimpse:backward({self.outputEncoder, self.outputDecoder}, gradLayer[1])

    gradDecoder:add(gradGlimpse[2]) -- accumulate gradient in-place

    self.decoder:backward(input[2], gradDecoder)

    -- initialize gradient from decoder
    local dhDec = self.decoder:get(2).gradHiddenInput
    local dhEnc = self.encoder:get(2).gradHiddenInput
    dhEnc:resizeAs(dhDec):copy(dhDec)

    local dcDec = self.decoder:get(2).gradCellInput
    local dcEnc = self.encoder:get(2).gradCellInput
    dcEnc:resizeAs(dcDec):copy(dcDec)

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
        local gradNorm = self.gradParams:norm()
        -- clip gradient
        if gradNorm > self.maxNorm then
            self.gradParams:mul(self.maxNorm / gradNorm)
        end
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
    self.outputEncoder = self.encoder:forward(x)
    self.hiddenOutput = self.encoder:get(2).hiddenOutput
    self.cellOutput = self.encoder:get(2).cellOutput
end

function NMT:stepDecoder(x)
    -- alias
    local hiddenOutput = self.decoder:get(2).hiddenOutput
    hiddenOutput:resizeAs(self.hiddenOutput):copy(self.hiddenOutput)

    local cellOutput = self.decoder:get(2).cellOutput
    cellOutput:resizeAs(self.cellOutput):copy(self.cellOutput)

    self.outputDecoder = self.decoder:forward(x)
    self.glimpseOutput = self.glimpse:forward({self.outputEncoder, self.outputDecoder})
    self.logProb = self.layer:forward{self.glimpseOutput, self.outputDecoder})

    self.hiddenOutput = self.decoder:get(2).hiddenOutput
    self.cellOutput = self.decoder:get(2).cellOutput

    return self.logProb
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
    self.buffers.prevState = self.buffers.prevState:index(2, index)
    if index:numel() ~= self.buffers.outputEncoder:size(1) then
        self.buffers.outputEncoder = self.buffers.outputEncoder:index(1, index)
    end
end

function NMT:repeatState(K)
    -- useful for beam search
    self.buffers.prevState = self.buffers.prevState:repeatTensor(1, K, 1)
    self.buffers.outputEncoder = self.buffers.outputEncoder:repeatTensor(K, 1, 1)
end

function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.layer:clearState()
    self.glimpse:clearState()
end
