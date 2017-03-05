require 'translErrors.localTranslError'
require 'nn'
require 'cutorch'
require 'cunn'

require 'data.loadBitext'
require 'tardis.confidentNMT'
require 'tardis.SeqAtt'
local _ = require 'moses'

local cfg = require 'pl.config'
local opt = cfg.read(arg[1])


local loader = DataLoader(opt)
opt.padIdx = loader.padIdx
local targetVocIdx = loader:getTargetIdx() -- TODO: check if correct implementation
local metaSymbols = {} -- loader:getMetaSymbolsIdx()


function prepro(input)
    local x, y = unpack(input)
    local seqlen = y:size(2)
    -- make contiguous and transfer to gpu
    x = x:contiguous():cudaLong()
    prev_y = y:narrow(2, 1, seqlen-1):contiguous():cudaLong()
    next_y = y:narrow(2, 2, seqlen-1):contiguous():cudaLong()

    return x, prev_y, next_y
end


--counters
local confidenceThresholds = {0.2,0.3,0.5,0.7,0.8,0.9,0.975}
local totalPredictions = 0
local totalCorrectPredictions = 0
local totalConfident = {}
local correctConfidenceScores = {}
local correctOK = {}
local correctBAD = {}
for _,threshold in ipairs(confidenceThresholds) do
    totalConfident[threshold] = 0
    correctConfidenceScores[threshold] = 0
    correctOK[threshold] = 0
    correctBAD[threshold] = 0
end

local confidenceIntervals = {{0.0,0.1},{0.1,0.2},{0.3,0.4},{0.4,0.5},{0.5,0.6},{0.6,0.7},{0.7,0.8},{0.8,0.9},{0.9,1.0}}
local confidenceScoreDistribution = {}
for _,interv in ipairs(confidenceIntervals) do
	confidenceScoreDistribution[interv] = {0,0} --BAD,OK
end

-- initialize model in evaluation mode
local model = nn.confidentNMT(opt)
model:type('torch.CudaTensor')
--model:load(opt.confidenceModelToLoad)
local pretrainedModel = nn.NMT(opt)
pretrainedModel:type('torch.CudaTensor')
pretrainedModel:load(opt.modelToLoad)
model:loadModelWithoutConfidence(pretrainedModel)
model:loadConfidence(opt.confidenceModelToLoad)

model:evaluate()
model.confidence:evaluate()

local errorCollection = {}
loader:loadForTesting(loader.tracker[1])
local nbatches = loader.nbatches


local beam = 20 
print("Number of batches: ",nbatches)
for i = 1, nbatches do
	local x, prev_y, next_y = prepro(loader:next_origOrder())
    model:clearState()
    model:forward({x,prev_y},next_y)
    local correctPredictions = model:extractCorrectPredictions(next_y,beam)
    --local predictedCorrectLosses = model:extractPairwiseLosses(next_y,beam)
    correctPredictions = correctPredictions:view(-1) 
    local cofidenceScores = model:extractConfidenceScores():double()
    totalPredictions = totalPredictions + correctPredictions:size(1)
    totalCorrectPredictions = totalCorrectPredictions + correctPredictions:sum()


    for j=1,correctPredictions:size(1) do
	local index = correctPredictions[j]+1
	local score = cofidenceScores[j][1]
	for interv,_ in pairs(confidenceScoreDistribution) do
		if score > interv[1] and score < interv[2] then
			confidenceScoreDistribution[interv][index] = confidenceScoreDistribution[interv][index] + 1 
		end
	end
    end
    for thr,counter in pairs(totalConfident) do
        local thresholdTensor = torch.Tensor(correctPredictions:size()):fill(thr)
        local confidentPredictions = torch.ge(cofidenceScores,thresholdTensor):view(-1)
	
        totalConfident[thr] = totalConfident[thr] + confidentPredictions:sum()
        local correctLabels = torch.eq(correctPredictions:float(),confidentPredictions:float())
	local numCorrectOK = 0
	for i=1,confidentPredictions:size(1) do
		--for j=1,confidentPredictions:size(2) do
			if confidentPredictions[i] == 1 then
				numCorrectOK = numCorrectOK + correctLabels[i]
			end
	end
	local numCorrectLabels = correctLabels:sum()
	local numCorrectBAD = numCorrectLabels - numCorrectOK
        correctConfidenceScores[thr] = correctConfidenceScores[thr] + numCorrectLabels
	correctOK[thr] = correctOK[thr] + numCorrectOK
	correctBAD[thr] = correctBAD[thr] + numCorrectBAD
    end
end

-- metrics:
--accuracy: how many of the predictted OK/BAD coinceded with label
--precision-OK: how many of OK-predicted were OK
--precision-BAD: how many of BAD-predicted were BAD
-- recall-OK: how many of OK were recovered
-- recall-BAD

print('total::'..totalPredictions)
print('total OK::'..totalCorrectPredictions..' '..totalCorrectPredictions/totalPredictions)
print('total BAD::'..(totalPredictions-totalCorrectPredictions)..' '..(1 - totalCorrectPredictions/totalPredictions))

for _,thr in ipairs(confidenceThresholds) do
    local acc = correctConfidenceScores[thr]/totalPredictions
    print('THRESHOLD '..thr..':: acc='..acc)
    local precOK = correctOK[thr]/totalConfident[thr]
    local precBAD = correctBAD[thr]/(totalPredictions - totalConfident[thr])
    local recOK = correctOK[thr]/totalCorrectPredictions
    local recBAD = correctBAD[thr]/(totalPredictions - totalCorrectPredictions)
    print('THRESHOLD '..thr..':: precOK='..precOK)
    print('THRESHOLD '..thr..':: precBAD='..precBAD)
    print('THRESHOLD '..thr..':: recallOK='..recOK)
    print('THRESHOLD '..thr..':: recallBAD='..recBAD)
end



for interv,counts in pairs(confidenceScoreDistribution) do
	local shareOK = counts[2]/totalCorrectPredictions
	local shareBAD = counts[1]/(totalPredictions-totalCorrectPredictions)
	print('INTERVAL')
	print(interv)
	print('OK '..shareOK..' BAD '..shareBAD)
end
