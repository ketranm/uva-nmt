-- author: Ke Tran <m.k.tran@uva.nl>
require 'cutorch'
require 'tardis.GlimpseDot'
require 'optim'
require 'tardis.FastTransducer'
require 'moses'
local model_utils = require 'tardis.model_utils'

local utils = require 'misc.utils'

local NMT, parent = torch.class('nn.NMT', 'nn.Module')

function NMT:__init(opt)
    -- build encoder
    local sourceSize = opt.sourceSize
    local inputSize = opt.inputSize
    local hiddenSize = opt.hiddenSize
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
    self.layer:add(nn.Tanh())
    self.layer:add(nn.Linear(hiddenSize, targetSize, true))
    self.layer:add(nn.LogSoftMax())

    local weights = torch.ones(targetSize)
    weights[opt.padIdx] = 0

    self.padIdx = opt.padIdx
    self.criterion = nn.ClassNLLCriterion(weights, true)

    self.maxNorm = opt.maxNorm or 5
    -- for optim
    self.optimConfig = {learningRate = 0.0002, beta1 = 0.9, beta2 = 0.999, learningRateDecay = 0.0001}
    self.optimStates = {}

    --self:reset()
end


function NMT:updateLearningRate(epoch)
end
function NMT:type(type)
    parent.type(self, type)
    self.params, self.gradParams =
        model_utils.combine_all_parameters(self.encoder,
                                           self.decoder,
                                           self.glimpse,
                                           self.layer)
end


function NMT:setType(type)
    parent.type(self, type)
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
    --self:extractCorrectPredictions(logProb,target)
    return self.criterion:forward(logProb, target)
end

-- author: Katya Garmash 
function NMT:forwardAndExtractErrors(input,target,errorFilters)
    self:stepEncoder(input[1])
    local logProb = self:stepDecoder(input[2])
    local predictions, errors = self:extractPredictionErrors(logProb,target)
    return predictions, errors
end

function NMT:extractCorrectPredictions(target,beam)
    if beam == nil then beam = 1 end
    local bla,predictions = self.logProb:topk(beam,true)
    if beam == 1 then 
	return torch.eq(predictions,target)
    else
	local target = target:view(-1)
	local result = torch.ByteTensor(target:size(1)):fill(0)
	for i=1,target:size(1) do
		local t = target[i]
		for j=1,predictions:size(2) do
			if predictions[i][j] == t then result[i] = 1 end
		end
	end
	return result
    end 
    
end


function NMT:extractPairwiseLoss(target)
    local topValues,predictions = self.logProb:topk(1,true)
    local pairwiseLosses = {} 
    for i=1,target:size(1) do
	table.insert(pairwiseLosses,topValues[i]-self.logProb[target[i][1]])
    end
    return pairwiseLosses
end
 
function NMT:backward(input, target)
    -- zero grad manually here
    self.gradParams:zero()
    local gradXent = self.criterion:backward(self.logProb, target:view(-1))
    --print(gradXent)
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

function NMT:optimize(input, target,mode)
    local feval = function(x)
        if self.params ~= x then
            self.params:copy(x)
        end
        local f = self:forward(input, target)
        self:backward(input, target,mode)
        local gradNorm = self.gradParams:norm()
        -- clip gradient
        if gradNorm > self.maxNorm then
            self.gradParams:mul(self.maxNorm / gradNorm)
        end
        return f, self.gradParams
    end
    local _, fx = optim.adam(feval, self.params, self.optimConfig, self.optimStates)
    return fx[1],self.confidLoss
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
--    self.decoder._rnn.rememberStates = false
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
    --print(self.prevStates:size())
    
    self.decoder:setStates(self.prevStates)
    self.decOutput = self.decoder:forward(x)
    self.cntx = self.glimpse:forward{self.encOutput, self.decOutput}
    self.logProb = self.layer:forward{self.cntx, self.decOutput}

    -- update prevStates
    self.prevStates = self.decoder:lastStates()
    return self.logProb
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
