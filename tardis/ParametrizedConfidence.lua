require 'tardis.PairwiseLoss'
require 'tardis.Confidence'
local utils = require 'misc.utils'
local Confidence, parent = torch.class('nn.ParametrizedConfidence', 'nn.Confidence')

function Confidence:__init(inputSize,hidSize,confidCriterion,opt)
    self.confidence = nn.Sequential()
    self.confidence:add(nn.Dropout(0.2))
    self.confidence:add(nn.Linear(inputSize,hidSize))
    self.confidence:add(nn.Tanh())
    self.confidence:add(nn.Dropout(0.2))
    local c = nn.ConcatTable()
    local objectiveMean = nn.Sequential()
    objectiveMean:add(nn.Linear(hidSize,1))
    objectiveMean:add(nn.Sigmoid())
    c:add(objectiveMean)
    local objectiveVar = = nn.Sequential()
    objectiveVar:add(nn.Linear(hidSize,1))
    objectiveVar:add(nn.Sigmoid())
    c:add(objectiveVar)
    self.confidence:add(c)
    
    self.downweightBAD = false 
    self.gradDownweight = 0.5
    self.good = 0
    self.total = 0
    self.downweightOK = false 

    self.correctBeam = opt.correctBeam
end

function Confidence:getObjectiveValue(mean,var,target)
	local diff = torch.csub(target,mean)
	local result = torch.cdiv(torch.cmul(diff,diff),var)
	result = result + torch.log(var)
	return torch.mul(result,1/2)
end

function Confidence:forward(inputState,logProb,target)
	local confidScore = self.confidence:forward(inputState)
	local confidMean = confidScore[1]
	local confidVar = confidScore[2]
	local loss = self:getObjectiveValue(confidMean,confidVar,target)
	self.confidMean = confidMean
	self.confidVar = confidVar
	return loss
end

function Confidence:computeConfidScore(inputState)
	local confidScore = self.confidence:forward(inputState)
	return confidScore[1],confidScore[2]
end




function Confidence:backward(inputState,target,logProb)
	local gradMean = torch.cmul(target - self.confidMean,1/self.confidVar) 
	local gradVar = 


   	
end







