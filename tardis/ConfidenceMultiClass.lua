require 'tardis.PairwiseLoss'
require 'tardis.topKDistribution'
local utils = require 'misc.utils'
local Confidence, parent = torch.class('nn.ConfidenceMultiClass', 'nn.Confidence')

function Confidence:__init(inputSize,hidSize,confidCriterion,classes,opt)
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
    self.confidence:add(nn.LogSoftMax())

    
    --self.confidence:add(nn.LogSoftMax())

    if confidCriterion == 'NLL' then
        self.confidenceCriterion = nn.ClassNLLCriterion()
        --self.confidenceCriterion = nn.ClassNLLCriterion(torch.ones(2),true)
    elseif confidCriterion == 'mixtureRandomGuessTopK' then
    	self.K = opt.K
    	self.confidenceCriterion = nn.ClassNLLCriterion(false,false)
    	self.matchingCriterion = nn.MSECriterion()
    	if opt.matchingObjective == 1 then
    		self.matchingObjective = 1
	else
		self.matchingObjective = 0
    	end
    elseif confidCriterion == 'mixtureCrossEnt' then
        self.confidenceCriterion = nn.ClassNLLCriterion()
    elseif confidCriterion == 'pairwise' then
	self.confidenceCriterion_1 = nn.PairwiseLoss(opt)
        self.confidenceCriterion_2 = nn.MSECriterion()
    end
    self.confidCriterionType = confidCriterion
    
    if opt.labelValue ~=nil then
    	self.labelValue = opt.labelValue
    else
    	self.labelValue = 'binary'
    end
    self.downweightBAD = false 
    self.gradDownweight = 0.5
    self.good = 0
    self.total = 0
    self.downweightOK = false 
    if opt.downweightOK == 1 then
	self.downweightOK = true 
     	self.gradDownweight = opt.gradDownweight 
    end
    self.correctBeam = opt.correctBeam
    if opt.labelValue ~=nil  then		
	self.labelValue = opt.labelValue
    else 
	self.labelValue = 'binary'
    end
end

function Confidence:updateCounts()
	for i=1,self.beamClasses:size(1) do
		self.classesCounts[self.beamClasses[i]] = self.classesCounts[self.beamClasses[i]]+1
	end
	self.total = self.total + self.beamClasses:size(1)
end

function Confidence:forwardLoss(confidScore,logProb,target)
    if self.confidCriterionType == 'NLL' then 
        local beamClasses = utils.extractBeamRegionOfCorrect(logProb,target,self.classes)
        self.confidLoss = {self.confidenceCriterion:forward(confidScore,beamClasses),0}
        self.beamClasses = correctPredictions:cuda()
		self:updateCounts()
    end 
end



function Confidence:backward(inputState,target,logProb)
   	local gradConfidCriterion = nil
	if self.confidCriterionType == 'NLL' then 
		gradConfidCriterion = self.confidenceCriterion:backward(self.confidScore,self.beamClasses)
	end
	local gradConfid = self.confidence:backward(inputState,gradConfidCriterion)
	return gradConfid
end

