require 'tardis.PairwiseLoss'
require 'tardis.topKDistribution'
local utils = require 'misc.utils'
local Confidence, parent = torch.class('nn.Confidence', 'nn.Module')

function Confidence:__init(inputSize,hidSize,confidCriterion,opt)
    self.confidence = nn.Sequential()
    self.confidence:add(nn.Linear(inputSize,hidSize))
    self.confidence:add(nn.Dropout(0.2))
    self.confidence:add(nn.Tanh())
    self.confidence:add(nn.Dropout(0.2))
    self.confidence:add(nn.Linear(hidSize,1))
    --self.confidence:add(nn.MulConstant(0.5))
    self.confidence:add(nn.Sigmoid())

    if confidCriterion == 'MSE' then
        self.confidenceCriterion = nn.MSECriterion()
    elseif confidCriterion == 'mixtureRandomGuessTopK' then
    	self.K = opt.K
    	self.confidenceCriterion = nn.ClassNLLCriterion(false,false)
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

function Confidence:load(modelFile)
	self.confidence = torch.load(modelFile)
end
function Confidence:clearState()
	self.confidence:clearState()
end

function Confidence:training()
	self.confidence:training()
end

function Confidence:clearStatistics()
	self.good = 0
	self.total = 0
end
function Confidence:evaluate()
	self.confidence:evaluate()
end

function Confidence:correctStatistics()
	local bad = self.total - self.good
	local shareGood = self.good/self.total
	local shareBad = bad/self.total
	return shareGood, shareBad
end
function Confidence:forward(inputState,logProb,target)
	local confidScore = self.confidence:forward(inputState)
	self:forwardLoss(confidScore,logProb,target)	
	self.confidScore = confidScore 
	return self.confidLoss
end

function Confidence:computeConfidScore(inputState)
	local confidScore = self.confidence:forward(inputState)
	return confidScore
end

function Confidence:getConfidScore()
	return self.confidScore
end

function Confidence:updateCounts()
	local total = self.correctPredictions:size(1) * self.correctPredictions:size(2)
	local good = self.correctPredictions:sum()
	self.good = self.good + good
	self.total = self.total + total
end

function Confidence:forwardLoss(confidScore,logProb,target)
    if self.confidCriterionType == 'MSE' then 
        local correctPredictions = utils.extractCorrectPredictions(logProb,target,self.labelValue)
        self.confidLoss = self.confidenceCriterion:forward(confidScore,correctPredictions)
        self.correctPredictions = correctPredictions:cuda()
		self:updateCounts()

	elseif self.confidCriterionType == 'mixtureRandomGuessTopK' then
		--self.unifKDistr,self.unifValue = topKUniform(logProb,self.K)
		self.unifValue = -1*torch.log(30000)
		self.unifKDistr = torch.CudaTensor(logProb:size()):fill(self.unifValue)
		self.confidMix = computeMixDistr(confidScore,logProb,self.unifKDistr)
		self.confidLoss = (self.confidenceCriterion:forward(self.confidMix,target))/logProb:size(1)
        
    elseif self.confidCriterionType == 'mixtureCrossEnt' then
        local oracleMixtureDistr = computeOracleMixtureDistr(confidScore,self.logProb,target)
        self.oracleMixtureDistr = oracleMixtureDistr
        self.confidLoss = self.confidenceCriterion:forward(oracleMixtureDistr,target)
    elseif self.confidCriterionType == 'pairwise' then
	--local confidLogProb = torch.add(logProb,torch.log(self.confidScore:expand(logProb:size())))
		local confidLoss_1 = self.confidenceCriterion_1:forward(confidScore,logProb,target)
        local correctPredictions = utils.extractCorrectPredictions(logProb,target,self.correctBeam)
        local confidLoss_2 = self.confidenceCriterion_2:forward(confidScore,correctPredictions)
	self.confidLoss = confidLoss_2
        self.correctPredictions = correctPredictions:cuda()
	self:updateCounts()
    end 
end


function computeMixDistr(weight,logProb1,logProb2)
    local logWeight = torch.log(weight):expandAs(logProb1) -- tensor
    local logSecondWeight = torch.log(1 - weight)
    local firstAdd = logProb1 + logWeight
    local diff = logSecondWeight:expand(logProb2:size()) + logProb2 - firstAdd
    local result = firstAdd + torch.log(torch.exp(diff) + 1)
    return result
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

function Confidence:backward(inputState,target,logProb)
   	local gradConfidCriterion = nil
	if self.confidCriterionType == 'MSE' then 
		gradConfidCriterion = self.confidenceCriterion:backward(self.confidScore,self.correctPredictions)
	elseif self.confidCriterionType == 'mixtureRandomGuessTopK' then
		gradConfidCriterion = torch.ones(target:size())
		--[[local topPr,ind = logProb:topk(self.K,true)
		for i=1,target:size(1) do
			local corrClass = target[i]
			for j=1,self.K do
				if ind[i][j] == corrClass then 
					gradConfidCriterion[i] = 1
					break
				end
			end
		end]]--

		local gradOutputDistr = self.confidenceCriterion:backward(self.confidMix,target:view(-1))
		local gradMixture = torch.cmul(gradOutputDistr,self.confidMix)	
		local gradLogprob = torch.cmul(gradOutputDistr,logProb) 
		self.confidMix = torch.exp(torch.sum(torch.mul(gradMixture,-1),2))
		gradConfidCriterion = gradConfidCriterion:cuda()
		gradConfidCriterion:cdiv(self.confidMix)
		local prCorr = torch.mul(torch.sum(gradLogprob,2),-1)
		local prUnif = torch.exp(self.unifValue)
		gradConfidCriterion:cmul(prCorr-prUnif)

	elseif self.confidCriterionType == 'mixtureCrossEnt' then
		gradConfidCriterion = self.confidenceCriterion:backward(self.oracleMixtureDistr,target:view(-1))
	elseif self.confidCriterionType == 'pairwise' then
		local gradConfidCriterion_1 = self.confidenceCriterion_1:backward(logProb)
		local gradConfidCriterion_2 = self.confidenceCriterion_2:backward(self.confidScore,self.correctPredictions)
		gradConfidCriterion = torch.mul(gradConfidCriterion_1,1)-- + gradConfidCriterion_2 
	end
	--print(gradConfidCriterion[1])
	if self.downweightBAD or self.downweightOK then
		local gradientWeights = self:computePerinstanceWeights()
		gradConfidCriterion:cmul(gradientWeights)
	end
		
	local gradConfid = self.confidence:backward(inputState,gradConfidCriterion)
	--print(self.correctPredictions[1])
	--print(gradConfid[1])
	return gradConfid
end

function Confidence:computePerinstanceWeights()
	local badInstances = self.correctPredictions:double():clone()
  	if self.downweightBAD then	
		badInstances = (-1)* badInstances:csub(torch.ones(self.correctPredictions:size()))
	end	
	local badInstancesWeights = torch.mul(badInstances,self.gradDownweight)
	if self.downweightBAD then
		return torch.add(self.correctPredictions:double(),badInstancesWeights):cuda()
	else
		return torch.add((-1)*badInstances:csub(torch.ones(self.correctPredictions:size())),badInstancesWeights):cuda()
	end
end









