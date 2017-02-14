require 'translErrors.localTranslError'
require 'nn'
require 'cutorch'
require 'cunn'

require 'data.loadBitext'
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
local confidenceThresholds = {0.7,0.8,0.9,0.975}
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


-- initialize model in evaluation mode
local model = nn.confidentNMT(opt)
model:type('torch.CudaTensor')
model:load(opt.modelFile)
model:evaluate()

local errorCollection = {}
loader:loadForTesting(loader.tracker[1])
local nbatches = loader.nbatches
print("Number of batches: ",nbatches)
for i = 1, nbatches do
	local x, prev_y, next_y = prepro(loader:next_origOrder())
    model:clearState()
    model:forward({x,prev_y},next_y)
    local correctPredictions = model:extractCorrectPredictions()
    local cofidenceScores = model:extractCOnfidenceScores()
    totalPredictions = totalPredictions + correctPredictions:size(1) * correctPredictions:size(2)
    totalCorrectPredictions = totalCorrectPredictions + correctPredictions:sum()
    for thr,counter for pairs(totalConfident) do
        local thresholdTensor = torch.Tensor(correctPredictions:size()):fill(thr)
        local confidentPredictions = torch.ge(cofidenceScores,thresholdTensor)
        totalConfident[thr] = totalConfident[thr] + confidentPredictions:sum()
        local correctLabels = torch.eq(correctPredictions,confidentPredictions)
        correctConfidenceScores[thr] = correctConfidenceScores[thr] + correctLabels:sum()
    end
end

-- metrics:
--accuracy: how many of the predictted OK/BAD coinceded with label
--precision-OK: how many of OK-predicted were OK
--precision-BAD: how many of BAD-predicted were BAD
-- recall-OK: how many of OK were recovered
-- recall-BAD


for _,thr in ipairs(confidenceThresholds) do
    local acc = correctConfidenceScores[thr]/totalPredictions
    print('THRESHOLD '..thr..':: acc='..acc..', prec-OK='..)
end


