
local utils = require 'misc.utils'
local Confidence, parent = torch.class('nn.Confidence', 'nn.Module')

function Confidence:__init__(inputSize,hidSize,confidCriterion)

	self.confidence = nn.Sequential()
    self.confidence:add(nn.Dropout(0.2))
    self.confidence:add(nn.Linear(inputSize,hidSize))
    self.confidence:add(nn.Tanh())
    self.confidence:add(nn.Dropout(0.2))
    self.confidence:add(nn.Linear(hidSize,1))
    self.confidence:add(nn.Sigmoid())

    if confidCriterion == 'MSE' then
        self.confidenceCriterion = nn.MSECriterion()
    elseif confidCriterion == 'mixtureCrossEnt' then
        self.confidenceCriterion = nn.ClassNLLCriterion()
    end
    self.confidCriterionType = confidCriterion
end

function Confidence:forward(inputState,logProb,target)

	self.confidScore = self.confidence:forward(inputState)
	self:forwardLoss(logProb,target)
	return self.confidLoss
end


function Confidence:forwardLoss(logProb,target)

	 if self.confidCriterionType == 'MSE' then 
        local correctPredictions = utils.extractCorrectPredictions(logProb,target)
        self.confidLoss = self.confidenceCriterion:forward(self.confidScore,correctPredictions)
        self.correctPredictions = correctPredictions:cuda()
        
    elseif self.confidCriterionType == 'mixtureCrossEnt' then
        local oracleMixtureDistr = computeOracleMixtureDistr(self.confidScore,self.logProb,target)
        self.oracleMixtureDistr = oracleMixtureDistr
        self.confidLoss = self.confidenceCriterion:forward(oracleMixtureDistr,target)
    end
    
end


function computeOracleMixtureDistr(weight,logProb,target)
    local logWeight = torch.log(weight):expandAs(logProb) -- tensor
    local logSecondWeight = torch.log(1 - weight)
    local result = logProb + logWeight
    for i=1,target:size(1) do
        local orig = result[i][target[i]]
        result[i][target[i]] = orig + torch.log(1+ torch.exp(logSecondWeight[i] - orig))
    end
    return result
end

function Confidence:backward(inputState,target)
   	local gradConfidCriterion = nil
	if self.confidCriterionType == 'MSE' then 
		gradConfidCriterion = self.confidenceCriterion:backward(self.confidScore,self.correctPredictions)
	elseif self.confidCriterionType == 'mixtureCrossEnt' then
		gradConfidCriterion = self.confidenceCriterion:backward(self.oracleMixtureDistr,target:view(-1))
	end
	
	local gradConfid = self.confidence:backward(inputState,gradConfidCriterion)
	return gradConfid
end












