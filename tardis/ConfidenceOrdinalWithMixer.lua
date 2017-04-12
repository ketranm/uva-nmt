require 'tardis.PairwiseLoss'
require 'tardis.topKDistribution'
require 'misc.LogProbMixtureTable'
require 'tardis.ensembleCombination'
local utils = require 'misc.utils'
local Confidence, parent = torch.class('nn.ConfidenceOrdinalWithMixer', 'nn.ConfidenceOrdinal')

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
    self.confidenceDecision = nn.Sequential() 
    self.confidenceDecision:add(nn.Linear(hidSize,#classes+1))
    self.activation = nn.LogSigmoid()

    --for smoothing distribution:
    self.mixtureTable = nn.LogProbMixtureTable()
    --to mix original and smoothing distributions:
    self:createMixer(inputSize,hidSize)

    
    --self.confidence:add(nn.LogSoftMax())
    self.maxK = opt.maxK
    if confidCriterion == 'NLL' then
        self.confidenceCriterion = nn.ClassNLLCriterion()
        --self.confidenceCriterion = nn.ClassNLLCriterion(torch.ones(2),true)
    end
    self.confidCriterionType = confidCriterion    
    if opt.multiObjectiveConfidence == 1 then
	self.backpropagateFromMixerToConfidenceClass = true
    else	
	self.backpropagateFromMixerToConfidenceClass = false
    end 
	
end

function Confidence:createMixer(inputSize,hidSize)
	local input = nn.ParallelTable()
	local experts = nn.ParallelTable()
	experts:add(nn.Identity())
	experts:add(nn.Identity())
    local interpolationGate = nn.Sequential()
    interpolationGate:add(nn.JoinTable(2,2))
    interpolationGate:add(nn.Linear(inputSize+hidSize,inputSize))
    interpolationGate:add(nn.Tanh())
    interpolationGate:add(nn.Linear(inputSize,2))
    interpolationGate:add(nn.LogSoftMax())
    input:add(interpolationGate)
    input:add(experts)

    self.mixer = nn.Sequential()
    self.mixer:add(input)
    self.mixer:add(nn.LogProbMixtureTable())
    self.mixerCriterion = nn.ClassNLLCriterion() 
end

function Confidence:loadConfidenceOrdinalWithoutMixer(modelFile)

function Confidence:forwardConfidScore(inputState)
	self.confidenceHidState = self.confidence:forward(inputState)
	local confidenceUnnorm = self.confidenceDecision:forward(self.confidenceHidState)
    local confidScore = self.activation:forward(confidenceUnnorm)
	return confidScore
end

function Confidence:forward(inputState,logProb,target)
	local confidScore = self:forwardConfidScore(inputState)
	self:forwardLoss(confidScore,logProb,target,inputState)
	self.confidScore = confidScore
	return self.confidLoss
end
	

function convertCumulativeToClassSpecific(confidScore) 
	local  probs = torch.exp(confidScore)
	local  numClasses = confidScore:size(2)-1
	local indices = {}
	for i=1,numClasses do	
		table.insert(indices,i)
	end
	local bla = 1-probs:index(2,torch.LongTensor(indices))
	local inverse = torch.cat({torch.ones(confidScore:size(1),1):cuda(),bla},2)
	local result = torch.log(torch.cmul(probs,inverse))
	return result
end
	
function Confidence:forwardLoss(confidScore,logProb,target,inputState)
    if self.confidCriterionType == 'NLL' then 
        local beamClasses = utils.extractBeamRegionOfCorrect(logProb,target,self.classes):cuda()
	    local classSpecificLogProb = convertCumulativeToClassSpecific(confidScore)
   	    local smoothingLogProb = self:computeSmoothingOutput(logProb,classSpecificLogProb)
	    self.classSpecificLogProb = classSpecificLogProb
	    self.smoothingLogProb = smoothingLogProb
	    self.smoothedInterpolation = self:computeSmoothedInterpolation(logProb,smoothingLogProb,inputState,self.confidenceHidState)
	    -- self.confidLoss = {self.confidenceCriterion:forward(classSpecificLogProb,beamClasses),0}
	    self.confidLoss = {self.confidenceCriterion:forward(classSpecificLogProb,beamClasses),self.mixerCriterion:forward(self.smoothedInterpolation,target:view(-1))}
        self.beamClasses = beamClasses:cuda()
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

function Confidence:clearState()
	self.confidence:clearState()
	self.experts = nil
        self.weightedExperts = nil
	self.classSpecificLogProb = nil 
        self.smoothingLogProb = nil 
        self.smoothedInterpolation = nil 
        self.beamClasses = nil
end 
	
function Confidence:updateParameters(opt)
     self.maxK = opt.maxK
    parent.updateParameters(self,opt)
   self.mixtureTable = nn.LogProbMixtureTable()
   self.mixtureTable:cuda()
   self.uniformSmoothingFunc = scalarCombination({0.5,0.5},30000,'prob')
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
        	gradConfidCriterion = (torch.exp(self.confidScore) - cumulativeTargets)
	end
	local gradMixer = self:backwardMixerGate(inputState,self.confidenceHidState,logProb,self.smoothingLogProb,self.smoothedInterpolation,target:view(-1))
	local gradDecision = self.confidenceDecision:backward(self.confidenceHidState,gradConfidCriterion)
	if self.backpropagateFromMixerToConfidenceClass then gradDecision = gradDecision + gradMixer end 
	local gradConfid = self.confidence:backward(inputState,gradDecision)
	return gradConfid
end


function Confidence:backwardMixerGate(decoderState,confidenceState,logProb,smoothingLogProb,interpolatedDistr,target)
	local gradCrit = self.mixerCriterion:backward(interpolatedDistr,target)
	local gradMixer = self.mixer:backward({{decoderState,confidenceState},{logProb,smoothingLogProb}},gradCrit)
	return gradMixer[1][2]
end
	

function Confidence:computeConfidScore(inputState)
        local confidScore = self.activation:forward(self.confidence:forward(inputState))
        
        return confidScore
end


function Confidence:computeUniformMix(inputState,logProb)
    local confidScore = self:computeConfidScore(inputState)
    confidScore = convertCumulativeToClassSpecific(confidScore)
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

function Confidence:computeSmoothingOutput(logProb,classSpecificLogprob)
    local experts = {}
    local prevClass = 0
    for _,cl in ipairs(self.classes) do
        table.insert(experts,uniformizeExpert_2(logProb,cl,prevClass))
        prevClass = cl
    end
    table.insert(experts,uniformizeExpert_2(logProb,self.maxK,prevClass))
    local weightedExperts = self.mixtureTable:forward({classSpecificLogprob,experts})
    return weightedExperts
end

function Confidence:computeSmoothedInterpolation(logProb,smoothingDistr,inputState,hiddenConfState)
   local smoothedDistr = self.mixer({{inputState,hiddenConfState},{logProb,smoothingDistr}}) 
   return smoothedDistr
end
