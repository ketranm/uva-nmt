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
    self.confidence:add(nn.Dropout(0.2))
    self.confidence:add(nn.Linear(hiddenSize,confidenceHidSize))
    self.confidence:add(nn.Tanh())
    self.confidence:add(nn.Dropout(0.2))
    self.confidence:add(nn.Linear(confidenceHidSize,1))
    self.confidence:add(nn.Sigmoid())
    self.confidenceCriterion = nn.MSECriterion()
    self.MSEweight = opt.MSEweight
    
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
    self.optimConfig = {learningRate = 0.0002, beta1 = 0.9, beta2 = 0.999, learningRateDecay = 0.0001}
    self.optimStates = {}

end


function NMT:loadModelWithoutConfidence(model)
    self.encoder= model.encoder
    self.decoder= model.decoder
    
    local hidLayer = nn.Sequential()
    hidLayer:add(model.layer:get(1):clone())
    hidLayer:add(model.layer:get(2):clone())
    hidLayer:add(model.layer:get(3):clone())
    hidLayer:add(model.layer:get(4):clone())
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
                                           self.confidence)
end

function NMT:load(fileName)
    local params = torch.load(fileName)
    self.params:copy(params)
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
    self:stepDecoderUpToHidden(input[2])
    self:predictTargetLabel()
    self:stepConfidencePred()
    
    local correctPredictions = self:extractCorrectPredictions(self.logProb,target)
    self.correctPredictions = correctPredictions:cuda()
    local mainLoss = self.criterion:forward(self.logProb,target)
    confidLoss = self.confidenceCriterion:forward(self.confidScore,self.correctPredictions)
    self.confidLoss = confidLoss
    return mainLoss,confidLoss 
end


function NMT:backward(input, target,mode)
    -- zero grad manually here
    self.gradParams:zero()
    if (mode == nil and self.trainingScenario == 'confidenceMechanism') or mode == 'confidence' then
    	local gradMSE = self.confidenceCriterion:backward(self.confidScore,self.correctPredictions)
    	local gradConfid = self.confidence:backward(self.hidLayerOutput,gradMSE)
    else
       local gradXent = self.criterion:backward(self.logProb, target:view(-1))
       local gradOutputLayer = self.outputLayer:backward(self.hidLayerOutput,gradXent)
	local gradHidLayer = nil
       if self.trainingScenario ~= 'confidenceMechanism' then
       		local multiObjectiveGrad = torch.mul(gradOutputLayer,self.NLLweight):add(torch.mul(gradConfid,self.MSEweight))
       		local gradHidLayer = self.hidLayer:backward({self.cntx, self.decOutput}, multiObjectiveGrad)
	else if mode == 'NMT' then
		local gradHidLayer = self.hidLayer:backward({self.cntx, self.decOutput}, gradOutputLayer)
	end
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


function NMT:stepDecoderUpToHidden(x)
    self.decoder:setStates(self.prevStates)
    self.decOutput = self.decoder:forward(x)
    self.cntx = self.glimpse:forward{self.encOutput, self.decOutput}
    self.hidLayerOutput = self.hidLayer:forward{self.cntx, self.decOutput}
    -- update prevStates
    self.prevStates = self.decoder:lastStates()
end


function NMT:stepConfidencePred()
    self.confidScore = self.confidence:forward(self.hidLayerOutput)
    return self.confidScore
end

function NMT:extractConfidenceScores()
    return self.confidScore
end

function NMT:predictTargetLabel()
    local logProb = self.outputLayer:forward(self.hidLayerOutput)
    self.logProb = logProb
    return self.logProb
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
	torch.save(filename,self.confidence)
    else
    torch.save(fileName, self.params)
    end
end
