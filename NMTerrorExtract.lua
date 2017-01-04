require 'translErrors.localTranslError'




local loader = DataLoader(opt)
opt.padIdx = loader.padIdx
local prepro = require 'main.prepro'
local targetVocIdx = loader:getTargetIdx() -- TODO: check if correct implementation
local metaSymbols = loader:getMetaSymbolsIdx()


-- initialize error extractors
local errorExtractors = {}
for _,a in ipairs(arg) do
	if a == 'BINARY' then
		errorExtractors[a] = localTranslError.LocalTranslError()
	elseif a == 'REF_SYMBOL' then
		errorExtractors[a] = localTranslError.RefWordTranslError(targetVocIdx,metaSymbols)
	else
		ngramOrder = string.match(a,"NGRAM([0-9]+)") then
		if not ngramOrder then
			--if errorExtractors["NGRAM"] == nil then errorExtractors["NGRAM"] = {} end
			errorExtractors[a] = 
				localTranslError.NgramContextTranslError(targetVocIdx,metaSymbols,tonumber(ngramOrder))
		end
	end
end



--extract errors from batch
function extractTypedErrors(predictions, reference, errorsPositions,errorExtractors)
	assert(errors:size() == predictions:size())
	local numSent = errors:size(1)
    local seqLen = errors:size(2)

    for i=1,numSent do
        local predVector = predictions[i]
        local refVector = reference[i]
        for j=1,seqLen do
        	if errors[i][j] then
        		for _,extractor in pairs(errorExtractors) do
        			extractor:extractError(i,j,refVector,predVector)
        		end
        	end
        end
    end
end

-- initialize model in evaluation mode
local model = nn.NMT(opt)
model:type('torch.CudaTensor')
model:load(opt.modelFile)
model:evaluate()


local nbatches = loader.nbatches
for i = 1, nbatches do
	local x, prev_y, next_y = prepro(loader:next())
    model:clearState()
    local predictions, errors = model:forwardAndExtractErrors({x,prev_y},next_y)
    extractTypedErrors(predictions,next_y, errors,errorExtractors)
    if i % opt.reportEvery == 0 then
    	print("Batch ",i)
    end
end

--save extracted observations for each error type separately
local saveFile = opt.errorsFile
for errorType,extractor in pairs(errorExtractors) do
	torch.save(string.format("opt.errorsFile_%s.dat",errorType),extractor.errorObservations)
end



