require 'translErrors.localTranslError'
require 'nn'
require 'cutorch'
require 'cunn'
local cfg = require 'pl.config'
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
errorExtractor_11 =torch.BinaryTranslError(errorInputSize)

--compute statistics on actual set
local reference = torch.load(opt_1.reference)
local numRefTensors = #reference
-- sys_1 and sys_2
for i=1,numRefTensors do 
	extractTypedErrors(errors_1[i]:double(),errors_2[i]:double(),reference[i][2],{errorExtractor_12}) 
	extractTypedErrors(errors_1[i]:double(),errors_1[i]:double(),reference[i][2],{errorExtractor_11}) 
end
local statistics_12 = errorExtractor_12:computePairwiseStatistics()
local statistics_11 = errorExtractor_11:computePairwiseStatistics()
errorExtractor_12:reset()




r = 5000 -- num of permutations
c_1 = {}
c_2 = {}
actual_stat_diff = {}
for stat,_ in pairs(statistics_12[1]) do
	c_1[stat] = 0 -- count of times random permuatation stats was higher than actual one
	c_2[stat] = 0
	actual_stat_diff[stat] = torch.abs(statistics_11[1][stat] - statistics_12[1][stat])
	print(stat)
	print(actual_stat_diff[stat])
end

local errorExtractors_1rand = torch.BinaryTranslError(errorInputSize)
local errorExtractors_2rand = torch.BinaryTranslError(errorInputSize)

for iter=1,r do
	print(iter)
	errorExtractors_rand:reset()	
	for i=1,numRefTensors do 
		local curr_1 = errors_1[i]
		local curr_2 = errors_2[i]
		for j=1,curr_1:size(1) do
			for k=1,curr_1:size(2) do
				local swap_1 = torch.random(0,1) 
				local swap_2 = torch.random(0,1) 
				if swap_1 == 1 then curr_1[j][k] = torch.abs(curr_1[j][k]-1)
				if swap_2 == 1 then curr_2[j][k] = torch.abs(curr_2[j][k]-1)
			end
		end
		extractTypedErrors(errors_1[i]:double(),curr_1:double(),reference[i][2],{errorExtractors_1rand}) 
		extractTypedErrors(errors_2[i]:double(),curr_2:double(),reference[i][2],{errorExtractors_2rand}) 
	end
	local r_statistics_1 = errorExtractors_1rand:computePairwiseStatistics()
	local r_statistics_2 = errorExtractors_2rand:computePairwiseStatistics()
	for stat,r_val_1 in pairs(r_statistics_1[1]) do
		local diff_1 = torch.abs(statistics_11[1] - r_val_1)
		if diff_1 >= actual_stat_diff[stat] then 
			c_1[stat] = c_1[stat] + 1
		end
		local diff_2 = torch.abs(statistics_11[1] - r_statistics_2[1][stat])
		if diff_2 >= actual_stat_diff[stat] then 
			c_2[stat] = c_2[stat] + 1
		end
	end
end


for stat,c1 in pairs(c_1) do
	local p_val_1 = c1/r
	print(stat..'__1::'..p_val_1)
	local p_val_2  = c_2[stat]/r
	print(stat..'__2::'..p_val_2)
end



