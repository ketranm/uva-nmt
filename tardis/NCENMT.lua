-- author: Ke Tran <m.k.tran@uva.nl>

require 'tardis.GlimpseDot'
require 'optim'
require 'tardis.FastTransducer'
require 'tardis.NCEModule'
require 'tardis.NCECriterion'
local model_utils = require 'tardis.model_utils'

local utils = require 'misc.utils'

local NMT, parent = torch.class('nn.NMT', 'nn.Module')

function NMT:__init(opt)
    -- build encoder
    local sourceSize = opt.sourceSize
    local inputSize = opt.inputSize
    local hiddenSize = opt.hiddenSize
    local k = opt.numNoise  or 50

    local unigrams = opt.unigrams
    unigrams:div(unigrams:sum())

    self.encoder = nn.Transducer(sourceSize, inputSize, hiddenSize, opt.numLayers, opt.dropout)

    -- build decoder
    local targetSize = opt.targetSize
    self.decoder = nn.Transducer(targetSize, inputSize, hiddenSize, opt.numLayers, opt.dropout)

    -- attention
    self.glimpse = nn.GlimpseDot()

    self.layer = nn.Sequential()
    self.layer:add(nn.JoinTable(3))
    self.layer:add(nn.View(-1, 2 * hiddenSize))
    self.layer:add(nn.Linear(2 * hiddenSize, hiddenSize, false))
    self.layer:add(nn.ReLU())

    self.ncem = nn.NCEModule(hiddenSize, targetSize, k, unigrams, 1)
    self.ncec = nn.NCECriterion()

    self.sizeAverage = true
    self.padIdx = opt.padIdx

    local weights = torch.ones(targetSize)
    weights[opt.padIdx] = 0
    self.criterion = nn.ClassNLLCriterion(weights, true)

    -- convert to cuda
    self.encoder:cuda()
    self.decoder:cuda()
    self.glimpse:cuda()
    self.layer:cuda()
    self.ncem:cuda()
    self.ncec:cuda()
    self.criterion:cuda()

    self.params, self.gradParams =
        model_utils.combine_all_parameters(self.encoder,
                                           self.decoder,
                                           self.glimpse,
                                           self.layer,
                                           self.ncem)
    self.maxNorm = opt.maxNorm or 5
    -- for optim
    self.optimConfig = {}
    self.optimStates = {}
    -- turn this flag on for testing
    self.normalized = false
    self.train = true
    --self:reset()
    self._maskw = torch.CudaTensor()
end

function NMT:reset()
    self.params:uniform(-0.01, 0.01)
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

    local htop = self:stepDecoder(input[2])
    local nll = 0
    if self.train == true then
        self.probs = self.ncem:forward{htop, target}
        nll = self.ncec:forward(self.probs, target)
    else
        nll = self.criterion:forward(self.logProb, target)
    end
    return nll
end

function NMT:backward(input, target)
    -- zero grad manually here
    self.gradParams:zero()
    local target = target:view(-1)
    local gradNCE = self.ncec:backward(self.probs, target)

    -- zero out gradient to padIdx
    self._maskw:ne(target, self.padIdx) -- masked padding
    gradNCE[1]:cmul(self._maskw)
    gradNCE[3]:cmul(self._maskw)
    local n, k = gradNCE[2]:size(1), gradNCE[2]:size(2)
    local maskn = self._maskw:view(-1, 1):expand(n, k)
    gradNCE[2]:cmul(maskn)
    gradNCE[4]:cmul(maskn)

    local gradhtop = self.ncem:backward({self.htop, target}, gradNCE)

    local gradLayer = self.layer:backward({self.cntx, self.decOutput}, gradhtop[1])

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
    self.train = true
    self.ncem.train = true
    self.ncem.normalized = false
    self.encoder:training()
    self.decoder:training()
end

function NMT:evaluate()
    self.train = false
    self.ncem.train = false
    self.ncem.normalized = true
    self.ncem.logsoftmax = true

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
    self.htop = self.layer:forward{self.cntx, self.decOutput}

    if self.train == false then
        self.logProb = self.ncem:forward({self.htop, torch.CudaTensor({0})})
    end

    -- update prevStates
    self.prevStates = self.decoder:lastStates()
    return self.htop
end

function NMT:indexStates(index)
    self.encOutput = self.encOutput:index(1, index)
    self.decoder:indexStates(index)
    self.prevStates = self.decoder:lastStates()
end

function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.layer:clearState()
    self.glimpse:clearState()
end
