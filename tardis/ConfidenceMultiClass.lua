require 'tardis.PairwiseLoss'
require 'tardis.topKDistribution'
require 'misc.LogProbMixtureTable'
local utils = require 'misc.utils'
local Confidence, parent = torch.class('nn.ConfidenceMultiClass', 'nn.Confidence')

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
    self.confidence:add(nn.LogSoftMax())

    
    --self.confidence:add(nn.LogSoftMax())

    if confidCriterion == 'NLL' then
        self.confidenceCriterion = nn.ClassNLLCriterion()
        --self.confidenceCriterion = nn.ClassNLLCriterion(torch.ones(2),true)
    elseif confidCriterion == 'mixtureRandomGuessTopK' then
    	self.maxK = opt.maxK
    	self.confidenceCriterion = nn.ClassNLLCriterion()--false,false)
    	self.mixtureTable = nn.LogProbMixtureTable()
    	
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

function Confidence:correctStatistics()
	local result = {}
	for class,count in pairs(self.classesCounts) do 
		result[class] = count/self.total
	end
	return result
end	
function Confidence:clearState()
	self.confidence:clearState()
	self.experts = nil
	self.weightedExperts = nil
end	

function Confidence:clearStatistics()
	self.total = 0
	for cl,count in pairs(self.classesCounts) do
		self.classesCounts[cl] = 0
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
        local beamClasses = utils.extractBeamRegionOfCorrect(logProb,target,self.classes):cuda()
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



function Confidence:backward(inputState,target,logProb)
   	local gradConfidCriterion = nil
	if self.confidCriterionType == 'NLL' then 
		gradConfidCriterion = self.confidenceCriterion:backward(self.confidScore,self.beamClasses)
	elseif self.confidCriterionType == 'mixtureRandomGuessTopK' then
		local mixtureCriterion = self.confidenceCriterion:backward(self.weightedExperts,target:view(-1))
		--print(mixtureCriterion)
		gradConfidCriterion = self.mixtureTable:backward({self.confidScore,self.experts},mixtureCriterion)	
		--print(gradConfidCriterion)
	end
	local gradConfid = self.confidence:backward(inputState,gradConfidCriterion[1])
	return gradConfid
end

