-- author: Ke Tran <m.k.tran@uva.nl>
require 'tardis.GlimpseDot'
require 'optim'
require 'tardis.FastTransducer'
require 'moses'
require 'tardis.SeqAtt'
local model_utils = require 'tardis.model_utils'

local utils = require 'misc.utils'

local NMT, parent = torch.class('nn.confidencePredicitonOnly', 'nn.confidentNMT')

function NMT:__init(opt)
    -- build encoder
    self.trainingScenario = opt.trainingScenario
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
    self.maxNorm = opt.maxNorm or 5
    -- for optim
    self.optimConfig = {learningRate = 0.0002, beta1 = 0.9, beta2 = 0.999, learningRateDecay = 0.0001}
    self.optimStates = {}
    
    --self:reset()
end


function NMT:loadModelWithoutConfidence(model)
    self.encoder= model.encoder
    print(self.encoder == model.encoder)
    self.decoder= model.decoder
    local hidLayer = nn.Sequential()
    hidLayer:add(model.layer:get(1):clone())
    hidLayer:add(model.layer:get(2):clone())
    hidLayer:add(model.layer:get(3):clone())
    hidLayer:add(model.layer:get(4):clone())
    self.hidLayer = hidLayer
end

function NMT:type(type)
    self:setType(type)
 --   parent.type(self, type)
    self.params, self.gradParams =
        model_utils.combine_all_parameters(self.encoder,
                                           self.decoder,
                                           self.glimpse,
                                           self.hidLayer,
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
    self:stepDecoder(input[2])
    self:predictMultiTask() 
    local correctPredictions = self:extractCorrectPredictions(self.logProb,target)
    self.correctPredictions = correctPredictions:cuda()
    local mainLoss = self.criterion:forward(self.logProb,target)
    confidLoss = self.confidenceCriterion:forward(self.confidScore,self.correctPredictions)
    self.confidLoss = confidLoss
    return mainLoss,self.confidLoss 
end


function NMT:backward(input, target)
    -- zero grad manually here
    self.gradParams:zero()
    if self.trainingScenario == 'confidenceMechanism' then
	local gradMSE = self.confidenceCriterion:backward(self.confidScore,self.correctPredictions):cuda()
	print(gradMSE:size())
	self.confidence:backward(self.hidLayerOuput,gradMSE)
    else
    if self.adaptive then 
    local gradXent = torch.mul(self.criterion:backward(self.logProb, target:view(-1)),self.NLLweight)
    local gradMSE = torch.mul(self.confidenceCriterion:backward(self.confidScore,self.correctPredictions),self.MSEweight)
    local gradMultiTask = self.hidToObjectives:backward(self.hidLayerOutput,{gradXent,gradMSE})
    local gradHidLayer = self.hidLayer:backward({self.cntx, self.decOutput}, gradMultiTask)

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
    else
--	local gradMSE = self.confidenceCriterion:backward(self.confidScore,self.correctPredictions)
--	self.hidToObjectives:get(2):backward(self.hidLayerOutput,gradMSE)
	
   	local gradXent = self.criterion:backward(self.logProb, target:view(-1))
	--print(gradXent)
	local outputLayerGrad = self.hidToObjectives:get(1):backward(self.hidLayerOutput,gradXent)
    	local gradHidLayer = self.hidLayer:backward({self.cntx, self.decOutput}, outputLayerGrad)
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
    end
end

-- useful interface for beam search
function NMT:stepEncoder(x)
    self.encOutput = self.encoder:forward(x)
    self.prevStates = self.encoder:lastStates()
    return self.encOutput
end

function NMT:stepDecoder(x)
    self:stepDecoderUpToHidden(x)
    self:stepConfidencePred() 
    return self.logProb
end

function NMT:stepDecoderUpToHidden(x)
    self.decoder:setStates(self.prevStates)
    self.decOutput = self.decoder:forward(x)
    self.cntx = self.glimpse:forward{self.encOutput, self.decOutput}
    self.hidLayerOutput = self.hidLayer:forward{self.cntx, self.decOutput}
    -- update prevStates
    self.prevStates = self.decoder:lastStates()
end


function NMT:stepConfidencePred()
    self.confidenceScore = self.confidence:forward(self.hidLayerOutput)
    return self.confidenceScore
end


function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.hidLayer:clearState()
    self.glimpse:clearState()
    self.confidence:clearState()
end
