require 'translErrors.localTranslError'
require 'nn'
require 'cutorch'
require 'cunn'
local cfg = require 'pl.config'
local hypothesis = 'absoluteDivergence'
local opt_1 = cfg.read(arg[1]) -- sys_1 and sys_2

function extractTypedErrors(errors1, errors2, reference,errorExtractors)
    assert(errors1:size(1) == errors2:size(1))
    assert(reference:size(1) == errors1:size(1))
    assert(errors1:size(2) == errors2:size(2))
    assert(reference:size(2) == errors1:size(2))

    local numSent = errors1:size(1)
    local seqLen = errors2:size(2)
    intersect = torch.eq(errors1,errors2):double()
    errors_00 = torch.lt(errors1,intersect):double()
    errors_11 = torch.lt(errors_00,intersect):double()
    errors_10 = torch.gt(errors1,errors2):double()
    errors_01 = torch.gt(errors2,errors1):double()
    err = torch.cat({errors_11:view(numSent,seqLen,1),errors_00:view(numSent,seqLen,1),
                    errors_10:view(numSent,seqLen,1),errors_01:view(numSent,seqLen,1)},
                    3) -- TODO: which dimension?
    for i=1,numSent do
        local refVector = reference[i]
        for j=1,seqLen do
            positError = err[i][j] -- 4-len vector
                for _,extractor in pairs(errorExtractors) do
                        extractor:extractError(positError,j,refVector)
                end
        end
    end
end
--error indices
errorFiles = {}
for eFile in string.gmatch(opt_1.ensembleErrorFiles,"[^%s]+") do
    table.insert(errorFiles,eFile)
end


local errors_1 = torch.load(errorFiles[1])
local errors_2 = torch.load(errorFiles[2])


errorInputSize = 4 -- N_11,N_00,N_10,N_01


errorExtractor_12 = torch.BinaryTranslError(errorInputSize)

--compute statistics on actual sets
local reference = torch.load(opt_1.reference)
local numRefTensors = #reference
local vocab = torch.load(opt_1.trgVocFile)
local targetVocIdx= {}
local targetVocWords = {}
for i,w in pairs(vocab[2].idx2word) do
	table.insert(targetVocIdx,i)
	targetVocWords[i] =w
end
--		-- sys_1 and sys_2
for i=1,numRefTensors do 
	extractTypedErrors(errors_1[i]:double(),errors_2[i]:double(),reference[i][2],{errorExtractor_12}) 
end
local statistics_12 = errorExtractor_12:computePairwiseStatistics()
errorExtractor_12:reset()
r = 5000 -- num of permutations
c = {}
actual_stat_diff = {}
for stat,_ in pairs(statistics_12[1]) do
	c[stat] = 0 -- count of times random permuatation stats was higher than actual one
	print(stat)
	print(statistics_12[1][stat])
end

local intersect = {}
for i=1,numRefTensors do
	local e_1 = errors_1[i]
	local e_2 = errors_2[i]
	local inter = torch.eq(e_1,e_2)
	intersect[i] = {}
	for j=1,inter:size(1) do
		for k=1,inter:size(2) do
			if inter[j][k] == 0 then table.insert(intersect[i],{j,k}) end
		end
	end
end 

for iter=1,r do
	print(iter)
	errorExtractor_12:reset()
	for i=1,numRefTensors do 
		local curr_1 = errors_1[i]:clone()
		local curr_2 = errors_2[i]:clone()
		for _,posit in ipairs(intersect[i]) do
			local swap = torch.random(0,1) 
			if swap == 1 then
				local j = posit[1]
				local k = posit[2] 
				curr_1[j][k] = torch.abs(curr_1[j][k]-1)
				curr_2[j][k] = torch.abs(curr_2[j][k]-1)
			end
		end
		extractTypedErrors(curr_1:double(),curr_2:double(),reference[i][2],{errorExtractor_12}) 
	end
	local r_statistics_12 = errorExtractor_12:computePairwiseStatistics()
	for stat,r_val_12 in pairs(r_statistics_12[1]) do
		if stat == 'disagreementMeasure' then
			if r_val_12 >= statistics_12[1][stat] then c[stat] = c[stat] + 1 end
		else
			print(stat)
			print(r_val_12)
			if r_val_12 <= statistics_12[1][stat] then c[stat] = c[stat] + 1 end
		end
	end
end


for stat,c in pairs(c) do
	local p_val = c/r
	print(stat..'::'..p_val)
end



