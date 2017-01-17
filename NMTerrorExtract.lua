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
-- initialize error extractors
local errorExtractors = {}
for i=2,#arg do
	a = arg[i] 
	if a == 'BINARY' then
		errorExtractors[a] = torch.BinaryTranslError()
	elseif a == 'REF_SYMBOL' then
		errorExtractors[a] = torch.RefWordTranslError(targetVocIdx,metaSymbols)
	else
		ngramOrder = string.match(a,"NGRAM([0-9]+)")
		if ngramOrder then
			--if errorExtractors["NGRAM"] == nil then errorExtractors["NGRAM"] = {} end
			errorExtractors[a] = 
				torch.NgramContextTranslError(targetVocIdx,metaSymbols,tonumber(ngramOrder))
		end
	end
end




-- initialize model in evaluation mode
local model = nn.NMT(opt)
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
    local predictions, errors = model:forwardAndExtractErrors({x,prev_y},next_y)
    table.insert(errorCollection,errors)
    --extractTypedErrors(predictions,next_y, errors,errorExtractors)
    if i % opt.reportEvery == 0 then
    	print("Batch ",i)
    end
end

--save extracted observations for each error type separately
local saveFile = string.format("%s/%s",opt.dataPath,opt.errorsFile)

torch.save(saveFile,errorCollection)
print("Saved to ",daveFile)
--[[for errorType,extractor in pairs(errorExtractors) do
	print("Saving ",errorType) 
	torch.save(string.format("%s/%s_%s.dat",opt.dataPath,opt.errorsFile,errorType),extractor.errorObservations)
end]]--



