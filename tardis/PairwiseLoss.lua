--local logSumExp2 = require 'tardis.ensembleCombination.logSumExp2'
local PairwiseLoss, parent = torch.class('nn.PairwiseLoss','nn.Criterion')

function PairwiseLoss:__init(opt)
	self.pairwiseDiff = 'arithm'
end

function logNegExp2(vector)
	local vStar = vector[{{},{1}}]
	local v2 = vector[{{},{2}}]-vStar
	local result = vStar + torch.log(1 -  torch.exp(v2))
	for i=1,vector:size(1) do
		if vector[i][1] == vector[i][2] then result[i][1] = torch.log(10^(-60)) end
	end
	return result
end


function PairwiseLoss:forward(confidScore,logProb,target) -- assume logProb already multiplied by confidence
    local bestValues,bestPredictions = logProb:topk(1,true)
    local target = target:view(-1):float()
    local correctScores = torch.FloatTensor(target:size(1))
    for i=1,target:size(1) do
	local corr = target[i]
	correctScores[i] = logProb[i][corr]
    end
    --bestValues = torch.add(bestValues,torch.log(confidScore:expand(bestValues:size()))):view(-1)
    self.correctScores = correctScores 
    if self.pairwiseDiff == 'arithm' then 
	--print('----')
	--print(bestValues[1])
	--print(correctScores[1])
	local diff = logNegExp2(torch.FloatTensor.cat(bestValues:float(),correctScores))
	--print(diff[1])
	local weightedDiff = torch.exp(torch.add(diff,torch.log(confidScore:float():expand(diff:size()))))
	--print(weightedDiff[1])
	--local expDiff = torch.exp((-1)*weightedDiff)
	-- print(expDiff[1])
	return weightedDiff:sum()/diff:size(1)
    end 
--	bestValues[i][1] = bestValues[i][1] - correctScores[i] 
--    elf.correctScores = correctScores:cuda()
--    return bestValues:sum()/bestValues:size(1)
end



function PairwiseLoss:backward(logProb)
    local bestValues,bestPredictions = logProb:topk(1,true)
    if self.pairwiseDiff == 'arithm' then
	local diff = logNegExp2(torch.FloatTensor.cat({bestValues:float(),self.correctScores:view(-1,1)}))
	
        return torch.exp(diff):cuda()
    end
    --bestValues = bestValues:view(-1):csub(self.correctScores)
    --return bestValues
end


