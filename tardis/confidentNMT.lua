-- author: Ke Tran <m.k.tran@uva.nl>
require 'tardis.GlimpseDot'
require 'optim'
require 'tardis.FastTransducer'
require 'moses'
require 'tardis.SeqAtt'
local model_utils = require 'tardis.model_utils'

local utils = require 'misc.utils'

local NMT, parent = torch.class('nn.confidentNMT', 'nn.NMT')

function NMT:__init(opt)
    -- build encoder
    local sourceSize = opt.sourceSize
    local inputSize = opt.inputSize
    local hiddenSize = opt.hiddenSize
    local confidenceHidSize = opt.confidenceHidSize 
    self.encoder = nn.Transducer(sourceSize, inputSize, hiddenSize, opt.numLayers, opt.dropout)

    -- build decoder
    local targetSize = opt.targetSize
    self.decoder = nn.Transducer(targetSize, inputSize, hiddenSize, opt.numLayers, opt.dropout)

    -- attention
    self.glimpse = nn.GlimpseDot()
    --hid-tilda layer
    self.hidLayer = nn.Sequential()
    self.hidLayer:add(nn.JoinTable(3))
    self.hidLayer:add(nn.View(-1, 2 * hiddenSize))
    self.hidLayer:add(nn.Linear(2 * hiddenSize, hiddenSize, false))
    self.hidLayer:add(nn.Tanh())

    self.outputLayer = nn.Sequential()
    self.outputLayer:add(nn.Linear(hiddenSize, targetSize, true))
    self.outputLayer:add(nn.LogSoftMax())


    self.confidence = nn.Sequential()
    self.confidence:add(nn.Linear(hiddenSize,confidenceHidSize))
    self.confidence:add(nn.Tanh())
    self.confidence:add(nn.Linear(confidenceHidSize,1))
    self.confidence:add(nn.Sigmoid())
    self.confidenceCriterion = nn.MSECriterion()
    self.MSEweight = opt.MSEweight

    local weights = torch.ones(targetSize)
    weights[opt.padIdx] = 0

    self.padIdx = opt.padIdx
    self.criterion = nn.ClassNLLCriterion(weights, true)
    self.NLLweight = opt.NLLweight
    self.maxNorm = opt.maxNorm or 5
    -- for optim
    self.optimConfig = {}
    self.optimStates = {}

    --self:reset()
end

function NMT:type(type)
    self:setType(type)
 --   parent.type(self, type)
    self.params, self.gradParams =
        model_utils.combine_all_parameters(self.encoder,
                                           self.decoder,
                                           self.glimpse,
                                           self.hidLayer,
                                           self.outputLayer,
                                           self.confidence)
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

    local logProb = self:stepDecoder(input[2]) -- TODO
    local confidScore = self:stepConfidencePred() --TODO

    local _,errors = self:extractPredictionErrors(target)
    self.errors = errors:cuda()
    self.confidScore = confidScore
    local mainLoss =  self.criterion:forward(logProb, target)
    local confidLoss = self.confidenceCriterion:forward(confidScore,self.errors)
    return mainLoss, confidLoss, self.NLLweight * mainLoss + self.MSEweight * confidLoss
end


function NMT:backward(input, target)
    -- zero grad manually here
    print(target:size())
    self.gradParams:zero()
    local gradXent = torch.mul(self.criterion:backward(self.logProb, target:view(-1)),self.NLLweight)
    --print(type(gradXent))
    local gradMSE = torch.mul(self.confidenceCriterion:backward(self.confidScore,self.errors),self.MSEweight)
    print(gradMSE:size())
    local gradOutputLayer = self.outputLayer:backward(self.hidLayerOutput,gradXent)
    local gradConfid = self.confidence:backward(self.hidLayerOutput,gradMSE)    

    local gradHidLayer = self.hidLayer:backward({self.cntx, self.decOutput}, gradOutputLayer+gradConfid)

    local gradDecoder = gradHidLayer[2] -- grad to decoder
    local gradGlimpse =
        self.glimpse:backward({self.encOutput, self.decOutput}, gradHidLayer[1])

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
    self.hidLayerOutput = self.hidLayer:forward{self.cntx, self.decOutput}
    self.logProb = self.outputLayer:forward(self.hidLayerOutput)

    -- update prevStates
    self.prevStates = self.decoder:lastStates()
    return self.logProb
end

function NMT:stepConfidencePred()
    self.confidenceScore = self.confidence:forward(self.hidLayerOutput)
    return self.confidenceScore
end

function NMT:indexStates(index)
    self.encOutput = self.encOutput:index(1, index)
    self.decoder:indexStates(index)
    self.prevStates = self.decoder:lastStates()
end

function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.hidLayer:clearState()
    self.outputLayer:clearState()
    self.glimpse:clearState()
    self.confidence:clearState()
end
