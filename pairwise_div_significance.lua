require 'translErrors.localTranslError'
require 'nn'
require 'cutorch'
require 'cunn'
require 'computePairwiseDiversity'
local cfg = require 'pl.config'
local opt_1 = cfg.read(arg[1]) -- sys_1 and sys_2
local opt_2 = cfg.read(arg[1]) -- sys_1 and sys_3

--error indices
errorFiles = {}
for eFile in string.gmatch(opt_1.ensembleErrorFiles,"[^%s]+") do
    table.insert(errorFiles,eFile)
end
i = 1
for eFile in string.gmatch(opt_2.ensembleErrorFiles,"[^%s]+") do
	if i == 2 then table.insert(errorFiles,eFile) end
	i = i + 1 
end

errors_1 = torch.load(errorFiles[1])
errors_2 = torch.load(errorFiles[2])
errors_3 = torch.load(errorFiles[3])

errorInputSize = 4 -- N_11,N_00,N_10,N_01
errorExtractor_12 = torch.BinaryTranslError(errorInputSize)}
errorExtractor_13 = torch.BinaryTranslError(errorInputSize)

--compute statistics on actual sets
local reference = torch.load(opt_1.reference)
local numRefTensors = #reference
-- sys_1 and sys_2
for i=1,numRefTensors do 
	extractTypedErrors(errors_1[i]:double(),errors_2[i]:double(),reference[i][2],{errorExtractor_12}) 
	extractTypedErrors(errors_1[i]:double(),errors_3[i]:double(),reference[i][2],{errorExtractor_13}) 
end
local statistics_12 = errorExtractor_12:computePairwiseStatistics()
local statistics_13 = errorExtractor_13:computePairwiseStatistics()
errorExtractor_12:reset()
errorExtractor_13:reset()



r = 10000 -- num of permutations
c = {}
actual_stat_diff = {}
for stat,_ in pairs(statistics_12) do
	c[stat] = 0 -- count of times random permuatation stats was higher than actual one
	actual_stat_diff[stat] = statistics_12[stat] - statistics_13[stat]
end

for i=1,r do
	errorExtractor_12:reset()
	errorExtractor_13:reset()
	for i=1,numRefTensors do 
		curr_2 = errors_2[i]
		curr_3 = errors_3[i]
		local intersect = torch.eq(curr_2,curr_3)
		for j = intersect:size(1) do
			for k = intersect:size(1) do			
				if intersect[j][k] == 0 then 
					local swap = torch.random(0,1) 
					if swap == 1 then 
						curr_2[j][k] = torch.abs(curr_2[j][k]-1)
						curr_3[j][k] = torch.abs(curr_3[j][k]-1)
					end
				end
			end
		end
		extractTypedErrors(errors_1[i]:double(),curr_2:double(),reference[i][2],{errorExtractor_12}) 
		extractTypedErrors(errors_1[i]:double(),curr_3:double(),reference[i][2],{errorExtractor_13}) 
	end
	local r_statistics_12 = errorExtractor_12:computePairwiseStatistics()
	local r_statistics_13 = errorExtractor_13:computePairwiseStatistics()
	for stat,r_val_12 in pairs(r_statistics_12) do
		local diff = r_val_12 - r_statistics_13[stat]
		if diff >= actual_stat_diff[stat] then c[stat] = c[stat] + 1 end
	end
end


for stat,c in pairs(c) do
	local p_val = c/r
	print(stat..'::'..p_val)
end



