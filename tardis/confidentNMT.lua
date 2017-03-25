-- author: Ke Tran <m.k.tran@uva.nl>
require 'tardis.GlimpseDot'
require 'optim'
require 'tardis.FastTransducer'
require 'moses'
require 'tardis.SeqAtt'
require 'tardis.Confidence'
local model_utils = require 'tardis.model_utils'

local utils = require 'misc.utils'

local NMT, parent = torch.class('nn.confidentNMT', 'nn.NMT')

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
	
    self.confidence= nn.Confidence(hiddenSize,confidenceHidSize,opt.confidCriterion,opt)
    self.confidWeight = opt.confidWeight
    
    self.outputLayer = nn.Sequential()
    self.outputLayer:add(nn.Linear(hiddenSize, targetSize, true))
    self.outputLayer:add(nn.LogSoftMax())
    self.criterion = nn.ClassNLLCriterion(weights, true)
    self.NLLweight = opt.NLLweight

    local weights = torch.ones(targetSize)
    weights[opt.padIdx] = 0

    self.padIdx = opt.padIdx
    self.maxNorm = opt.maxNorm or 5
    -- for optim
    self.optimConfig = {learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, learningRateDecay = 0.0001}
    self.optimStates = {}



end
function NMT:correctStatistics()
	return self.confidence:correctStatistics()
end

function NMT:loadConfidence(modelFile)
    self.confidence = torch.load(modelFile)
end

function NMT:loadModelWithoutConfidence(model)
    self.encoder= model.encoder
    self.decoder= model.decoder
    
    local hidLayer = nn.Sequential()
    hidLayer:add(model.layer:get(1))
    hidLayer:add(model.layer:get(2))
    hidLayer:add(model.layer:get(3))
    hidLayer:add(model.layer:get(4))
    self.hidLayer = hidLayer

    local outputLayer = nn.Sequential()
    outputLayer:add(model.layer:get(5):clone())
    outputLayer:add(model.layer:get(6):clone())
    self.outputLayer = outputLayer
end

function NMT:type(type)
    self:setType(type) 
    self.params, self.gradParams =
        model_utils.combine_all_parameters(self.encoder,
                                           self.decoder,
                                           self.glimpse,
                                           self.hidLayer,
                                           self.outputLayer,
                                           self.confidence.confidence)
end

function NMT:load(fileName)
    local params = torch.load(fileName)
    self.params:copy(params)
end
    
function NMT:forward(input, target)
    local target = target:view(-1)
    self:stepEncoder(input[1])
    self:stepDecoderUpToHidden(input[2])
    self:predictTargetLabel()
    local mainLoss = self.criterion:forward(self.logProb,target)
    local confidLoss = self.confidence:forward(self.hidLayerOutput,self.logProb,target)
    self.confidLoss = confidLoss
    return mainLoss,self.confidLoss 
end

function NMT:predictConfidenceScore()
    return self.confidence:computeConfidScore(self.hidLayerOutput)
end

function NMT:backward(input, target,mode)
    -- zero grad manually here
    self.gradParams:zero()
    local hidLayerGradOutput = torch.CudaTensor() 
    if self.trainingScenario == 'confidenceMechanism' or (self.trainingScenario == 'alternating' and mode == 'confidence') then
        hidLayerGradOutput = self.confidence:backward(self.hidLayerOutput,target,self.logProb)
    elseif self.trainingScenario == 'alternating' and mode == 'NMT' then
        local gradXent = self.criterion:backward(self.logProb, target:view(-1))
        hidLayerGradOutput = self.outputLayer:backward(self.hidLayerOutput,gradXent)
    elseif self.trainingScenario == 'joint' then
        local gradXent = self.criterion:backward(self.logProb, target:view(-1))
	local gradConfid = self.confidence:backward(self.hidLayerOutput,target)
        local gradOutputLayer = self.outputLayer:backward(self.hidLayerOutput,gradXent)
	hidLayerGradOutput =  torch.mul(gradOutputLayer,self.NLLweight):add(torch.mul(gradConfid,self.confidWeight))
    end
	
        
    --backpropagate further if we dont just train the confidence mechanism
    if self.trainingScenario == 'alternating' or self.trainingScenario == 'joint' then
        local gradHidLayer = self.hidLayer:backward({self.cntx, self.decOutput}, hidLayerGradOutput)
        local gradDecoder = gradHidLayer[2]-- grad to decoder
        local gradGlimpse =
            self.glimpse:backward({self.encOutput, self.decOutput}, gradHidLayer[1])
        gradDecoder:add(gradGlimpse[2]) -- accumulate gradient in-placep
        self.decoder:backward(input[2], gradDecoder)
        -- initialize gradient from decoder
        local gradStates = self.decoder:gradStates()
        self.encoder:setGradStates(gradStates)
        -- backward to encoder
        local gradEncoder = gradGlimpse[1]
        self.encoder:backward(input[1], gradEncoder)
    end
	
end


function NMT:stepDecoderUpToHidden(x)
    self.decoder:setStates(self.prevStates)
    self.decOutput = self.decoder:forward(x)
    self.cntx = self.glimpse:forward{self.encOutput, self.decOutput}
    self.hidLayerOutput = self.hidLayer:forward{self.cntx, self.decOutput}
    -- update prevStates
    self.prevStates = self.decoder:lastStates()
end


function NMT:extractConfidenceScores()
    return self.confidence.confidScore
end

function NMT:predictTargetLabel()
    local logProb = self.outputLayer:forward(self.hidLayerOutput)
    self.logProb = logProb
    return self.logProb
end

function NMT:stepDecoder(x)
	self:stepDecoderUpToHidden(x)
	return self:predictTargetLabel()
end


function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.hidLayer:clearState()
    self.outputLayer:clearState()
    self.glimpse:clearState()
    self.confidence:clearState()
    self.logProb = nil
    self.confidScore = nil
    self.hidLayerOutput = nil
end

function NMT:save(fileName)
    if self.trainingScenario == 'confidenceMechanism' then
	torch.save(fileName,self.confidence)
    else
    torch.save(fileName, self.params)
    end
end
