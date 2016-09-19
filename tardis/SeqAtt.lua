-- author: Ke Tran <m.k.tran@uva.nl>

require 'tardis.GlimpseDot'
require 'optim'
require 'tardis.Transducer'
local model_utils = require 'tardis.model_utils'

local utils = require 'misc.utils'

local NMT, parent = torch.class('nn.NMT', 'nn.Module')

function NMT:__init(opt)
    -- build encoder
    local sourceSize = opt.sourceSize
    local inputSize = opt.embeddingSize
    local hiddenSize = opt.hiddenSize
    self.encoder = nn.Transducer(sourceSize, inputSize, hiddenSize, opt.numLayers, opt.dropout)

    -- build decoder
    local targetSize = opt.targetSize
    self.decoder = nn.Transducer(targetSize, inputSize, hiddenSize, opt.numLayers, opt.dropout)

    -- attention
    self.glimpse = nn.GlimpseDot(hiddenSize)

    self.layer = nn.Sequential()
    self.layer:add(nn.JoinTable(3))
    self.layer:add(nn.View(-1, 2 * hiddenSize))
    self.layer:add(nn.Linear(2 * hiddenSize, hiddenSize, false))
    self.layer:add(nn.Tanh())
    self.layer:add(nn.Linear(hiddenSize, targetSize, true))
    self.layer:add(nn.LogSoftMax())

    local weights = torch.ones(targetSize)
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

    --self:reset()
end

function NMT:reset()
    self.params:uniform(-0.1, 0.1)
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
    local gradXent = self.criterion:backward(self.logProb, target:view(-1))
    local scale = 1 / (self.sizeAverage and self.numSamples or 1)
    gradXent:mul(scale)

    local gradLayer = self.layer:backward({self.cntx, self.decOutput}, gradXent)

    local gradDecoder = gradLayer[2] -- grad to decoder
    local gradGlimpse =
        self.glimpse:backward({self.encOutput, self.decOutput}, gradLayer[1])

    gradDecoder:add(gradGlimpse[2]) -- accumulate gradient in-place

    self.decoder:backward(input[2], gradDecoder)

    -- initialize gradient from decoder
    local gradStates = self.decoder:gradStates()
    self.encoder:setGradStates(gradStates)
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
    --[[ Encode the source sequence
    Parameters:
    - `x` : source tensor, can be a matrix (batch)
    --]]
    self.encOutput = self.encoder:forward(x)
    self.prevStates = self.encoder:lastStates()
    return self.encOutput
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
    self.decoder:setStates(self.prevStates)
    self.decOutput = self.decoder:forward(x)
    self.cntx = self.glimpse:forward{self.encOutput, self.decOutput}
    self.logProb = self.layer:forward{self.cntx, self.decOutput}

    -- update prevStates
    self.prevStates = self.decoder:lastStates()
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
    self.decoder:indexStates(index)
end

function NMT:repeatEncoderOutput(K)
    -- useful for beam search
    self.encOutput  = self.encOutput:repeatTensor(1, K, 1)
end

function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.layer:clearState()
    self.glimpse:clearState()
end
