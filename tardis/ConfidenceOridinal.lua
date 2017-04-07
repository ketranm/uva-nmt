require 'tardis.PairwiseLoss'
require 'tardis.topKDistribution'
require 'misc.LogProbMixtureTable'
require 'tardis.ensembleCombination'
local utils = require 'misc.utils'
local Confidence, parent = torch.class('nn.ConfidenceOrdinal', 'nn.Confidence')

function Confidence:__init(inputSize,hidSize,classes,confidCriterion,opt)
    local num_hid = opt.num_hid
    self.classes = classes
    self.classesCounts  =  {}  -- table of the form {1,20,100} means top-1,top-20 without 1, top-100 without top-20,rest (outside of top 100)
    for i,_ in ipairs(classes) do self.classesCounts[i] = 0 end

    self.classesCounts[#classes+1] = 0
    self.confidence = nn.Sequential()
    self.confidence:add(nn.Dropout(0.2))
    self.confidence:add(nn.Linear(inputSize,hidSize))
    self.confidence:add(nn.Tanh())
    self.confidence:add(nn.Dropout(0.2))
    if num_hid == 2 then
    	self.confidence:add(nn.Linear(hidSize,hidSize))
    	self.confidence:add(nn.Tanh())
    elseif num_hid == 3 then
    	self.confidence:add(nn.Linear(hidSize,hidSize))
    	self.confidence:add(nn.Tanh())
	self.confidence:add(nn.Linear(hidSize,hidSize))
	self.confidence:add(nn.Tanh())
	self.confidence:add(nn.Dropout(0.2))
    end
    self.confidence:add(nn.Linear(hidSize,#classes+1))
    self.confidence:add(nn.Sigmoid())

    
    --self.confidence:add(nn.LogSoftMax())
    self.maxK = opt.maxK
    if confidCriterion == 'NLL' then
        self.confidenceCriterion = nn.ClassNLLCriterion()
        --self.confidenceCriterion = nn.ClassNLLCriterion(torch.ones(2),true)
    end
    
end






function Confidence:forwardLoss(confidScore,logProb,target)
    if self.confidCriterionType == 'NLL' then 
        local beamClasses = utils.extractBeamRegionOfCorrectOrdinal(logProb,target,self.classes):cuda()
        self.confidLoss = {self.confidenceCriterion:forward(confidScore,beamClasses),0}
        self.beamClasses = beamClasses:cuda()
		self:updateCounts()
	elseif self.confidCriterionType == 'mixtureRandomGuessTopK' then
		local experts = {}
		local prevClass = 0
		for _,cl in ipairs(self.classes) do
			table.insert(experts,uniformizeExpert_2(logProb,cl,prevClass))
			prevClass = cl
		end
		table.insert(experts,uniformizeExpert_2(logProb,self.maxK,prevClass))
		self.experts = experts
		local weightedExperts = self.mixtureTable:forward({confidScore,experts})
		local loss = self.confidenceCriterion:forward(weightedExperts,target:view(-1))
		self.weightedExperts = weightedExperts 
		self.confidLoss = {loss,0}
		--self:updateCounts()
    end 
end

function Confidence:computePerinstanceWeights()
	local classVector = torch.Tensor(self.beamClasses:size()):fill(self.downweightClass):cuda()
	local weightVector = torch.Tensor(self.beamClasses:size()):fill(1):cuda()
	weightVector:csub(torch.mul(torch.eq(classVector,self.beamClasses),(1-self.gradDownweight)):cuda())
	return weightVector
end

	

function Confidence:backward(inputState,target,logProb)
   	local gradConfidCriterion = nil
	if self.confidCriterionType == 'NLL' then 
        local cumulativeTargets = turnIntoCumulativeTarget(self.beamClasses,#self.classes+1)
        gradConfidCriterion = torch.cmul(cumulativeTargets - self.confidScore,cumulativeTargets)
end
	local gradConfid = self.confidence:backward(inputState,gradConfidCriterion)--[1])
	return gradConfid
end

function Confidence:computeUniformMix(inputState,logProb)
    local confidScore = self:computeConfidScore(inputState)
    local experts = {}
    local prevClass = 0
    for _,cl in ipairs(self.classes) do
        table.insert(experts,uniformizeExpert_2(logProb,cl,prevClass))
        prevClass = cl
    end
    table.insert(experts,uniformizeExpert_2(logProb,self.maxK,prevClass))
    local weightedExperts = self.mixtureTable:forward({confidScore,experts})
    local final = self.uniformSmoothingFunc({weightedExperts,logProb})
    return final 

end

