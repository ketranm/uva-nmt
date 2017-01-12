require 'translErrors.localTranslError'
require 'nn'
require 'cutorch'
require 'cunn'



local cfg = require 'pl.config'
local opt = cfg.read(arg[1])

--vocabulary
local vocabFile = opt.trgVocFile
local vocab = torch.load(vocabFile)
local targetIdx = {}
for i,w in pairs(vocab[2].idx2word) do
    table.insert(targetIdx,i)
end
local metaSymbols = {} -- loader:getMetaSymbolsIdx()


-- initialize error extractors
local errorInputType = 'pairwise'
local errorExtractors = {}
for i=2,#arg do
	a = arg[i] 
	if a == 'BINARY' then
		errorExtractors[a] = torch.BinaryTranslError(errorInputType)
	elseif a == 'REF_SYMBOL' then
		errorExtractors[a] = torch.RefWordTranslError(targetVocIdx,metaSymbols,errorInputType)
	else
		ngramOrder = string.match(a,"NGRAM([0-9]+)")
		if ngramOrder then
			--if errorExtractors["NGRAM"] == nil then errorExtractors["NGRAM"] = {} end
			errorExtractors[a] = 
				torch.NgramContextTranslError(targetVocIdx,metaSymbols,tonumber(ngramOrder),errorInputType)
		end
	end
end

--individual error files
errorFiles = {}
for eFile in string.gmatch(opt.ensembleErrorFiles,"[^%s]+") do
    table.insert(errorFiles,eFile)
end

--reference
local loader = DataLoader(opt)

--extract pairwise errors
function extractTypedErrors(errors1, errors2, reference,errorExtractors)
    assert(errors1:size(1) == errors2:size(1))
    assert(reference:size(1) == errors1:size(1))
    assert(errors1:size(2) == errors2:size(2))
    assert(reference:size(2) == errors1:size(2))

	local numSent = errors1:size(1)
    local seqLen = errors2:size(2)
    errors_11 = torch.eq(errors1,errors2)
    errors_00 = torch.ones(numSent,seqLen) - error1 - errors2
    errors_10 = torch.eq(errors1,torch.ne(errors1,errors2))
    errors_01 = torch.eq(errors2,torch.ne(errors1,errors2))
    err = torch.cat({errors_11:view(numSent,seqLen,1),errors_00:view(numSent,seqLen,1),
                    errors_10:view(numSent,seqLen,1),errors_01:view(numSent,seqLen,1)},
                    3) -- TODO: which dimension?
    for i=1,numSent do
        local refVector = reference[i]
        for j=1,seqLen do
            positError = err[i][j] -- 4-len vector
        	for _,extractor in pairs(errorExtractors) do
        		extractor:extractError(positError,refVector)
        	end
        end
    end
    
    local result = {}
    for errExtractorType,extractor in pairs(errorExtractors) do
        result[errExtractorType] = extractor:computePairwiseStatistics()
    end
    return result
end



local pairwiseStatisticsSum = {}
for errorExtractorType,_ in pairs(errorExtractors) do
    pairwiseStatisticsSum[errExtractorType] = initializePairwiseStatistics()
end
--pairwise computation
local reference = torch.load(opt.reference)
local numRefTensors = #reference
local ensemble_size = #errorFiles
currFirst = 1
while currFirst < ensemble_size do
    currSecond = currFirst+1
    firstErrors = torch.load(errorFiles[currFirst])
    numTensors1 = #firstErrors
    while currSecond < ensemble_size+1 do
        intersection = {}
        secondErrors = torch.load(errorFiles[currSecond])
        numTensors2 = #secondErrors
        assert(numTensors1 == numRefTensors)
        assert(numTensors2 == numRefTensors)
        for _,eE in pairs(errorExtractors) do
            eE:reset()
        end
        for i=1,numRefTensors do
            extractTypedErrors(firstErrors[i],secondErrors[i],reference[i],errorExtractors)
        end

        for errorExtractorType,extractor in pairs(errorExtractors) do
            statistics = extractor:computePairwiseStatistics()
            for stat,val in pairs(statistics) do
                pairwiseStatisticsSum[errExtractorType][stat] = pairwiseStatisticsSum[errExtractorType][stat] + val
            end
        end
    end
end
--normalize averaged statistics
print("Ensemble: ")
print(opt.ensembleSet)
local normConst = 2/(ensemble_size*(ensemble_size-1))
for errorExtractorType,statSum in pairs(pairwiseStatisticsSum) do
    print("Averaged pairwise diversity measures for observation type "..errExtractorType)
    for stat,val in pairs(statSum) do
        statSum[stat] = val*normConst
        print(stat.." :: "..statSum[stat])
    end
    print("")
end


           
           