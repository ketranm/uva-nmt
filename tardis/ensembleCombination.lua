local _ = require 'moses'
require 'tardis.topKDistribution'
function scalarCombination(scWeights,outputDim,inputType)
	local weights = scWeights
	--print(weights)
	if inputType == 'prob' then
		print('test')
		local logWeights = _.map(weights,function(i,v) return torch.log(v) end)
		print(logWeights)
		function comb(tableOutputs)
			local len = tableOutputs[1]:size(1)
			--local weigtedExperts = _.map(tableOutputs, function(i,v) return v:add(logWeights[i]):view(len,outputDim,1) end)
			local weigtedExperts = _.map(tableOutputs, function(i,v) return v:add(logWeights[i]) end)
   			--[[local logProducts = torch.cat(weigtedExperts,3) -- has dim. (len,outputDim,2)
   			local maxLogs,_ = logProducts:topk(1,true)--:expandAs(logProducts)
   			
   			assert(maxLogs:size(1) == len)
   			assert(maxLogs:size(2) == outputDim)
   			assert(maxLogs:size(3) == 1)
   			local expSum = torch.CudaTensor(logProducts:size(1),logProducts:size(2))
   			for i=1,logProducts:size(3) do
   				local normalized = logProducts[{{},{},{i,i}}]-maxLogs
       			expSum:add(torch.exp(normalized))
   			end
   			return torch.log(expSum) + maxLogs]]--

   			local secondAdd  = torch.exp(weigtedExperts[2]-weigtedExperts[1])
   			local oneTensor = torch.CudaTensor(secondAdd:size()):fill(1.0)
   			secondAdd = torch.log(secondAdd+oneTensor)
   			return weigtedExperts[1] + secondAdd
		end
		return comb
	else
		function comb(tableOutputs)

			local totalOut = torch.mul(tableOutputs[1],weights[1])
			for i=2,#tableOutputs do
				local weighted = torch.mul(tableOutputs[i],weights[i])
				totalOut:add(weighted)
			end 

			return totalOut
		end
		
		return comb
	end
end

function logSumExp2(vector) 
	local vStar = vector[{{},{1}}]
	local v2 = vector[{{},{2}}]-vStar
	return vStar + torch.log(1 + torch.exp(v2))
end


function getTopKDistributions(tableOutputs,topK)
		local result = {}
		for _,d in ipairs(tableOutputs) do
			local vals,_ = d:topk(topK,true)
			table.insert(result,vals)
		end
		return result
end

function negEntropy(distrib)
	local prob = torch.exp(distrib)
	local result = torch.cmul(distrib,prob)
	return result
end

function entropyConfidence()
	local temperature = 1
	local topK = 5 
	
	function comb(tableOutputs)
		collectgarbage()
		local tableOutputs_topk = tableOutputs
		if topK > 0 then
			tableOutputs_topk =  getTopKDistributions(tableOutputs,topK)
		end
		local negEntropies = torch.cat(_.map(tableOutputs_topk,function(i,v) return negEntropy(v) end))
		negEntropies = negEntropies/temperature

		local logWeights = negEntropies - logSumExp2(negEntropies):expand(negEntropies:size())
		local weigtedExperts = _.map(tableOutputs, function(i,v) return v:add(logWeights[{{},{i}}]:expand(tableOutputs[1]:size())) end)
   		local secondAdd  = torch.exp(weigtedExperts[2]-weigtedExperts[1])
   		local oneTensor = torch.CudaTensor(secondAdd:size()):fill(1.0)
   		secondAdd = torch.log(secondAdd+oneTensor)
   		return weigtedExperts[1] + secondAdd
	end

	return comb
end

function entropyConfidenceBinaryReverse()
	local topK = 10

	function comb(tableOutputs)
		local tableOutputs_topk = tableOutputs
		if topK > 0 then
			tableOutputs_topk =  getTopKDistributions(tableOutputs,topK)
		end
		local negEntropies = torch.cat(_.map(tableOutputs_topk,function(i,v) return negEntropy(v) end))

		local result = tableOutputs[1]
		for i=1,negEntropies:size(1) do
			if negEntropies[1][i] >  negEntropies[2][i] then
				result[i] = tableOutputs[2][i]
			end
		end
		return result
	end
	return comb
end
function entropyConfidenceBinary()
	local topK = 10

	function comb(tableOutputs)
		local tableOutputs_topk = tableOutputs
		if topK > 0 then
			tableOutputs_topk =  getTopKDistributions(tableOutputs,topK)
		end
		local negEntropies = torch.cat(_.map(tableOutputs_topk,function(i,v) return negEntropy(v) end))

		local result = tableOutputs[1]
		for i=1,negEntropies:size(1) do
			if negEntropies[1][i] <  negEntropies[2][i] then
				result[i] = tableOutputs[2][i]
			end
		end
		return result
	end
	return comb
end

function confidenceInterpolation(direction)
	local direction = direction
	function comb(tableOutputs,confidScores)
		--print(confidScores[1][1])
		local logCombinWeights = torch.log(confidScores[1])--:expandAs(tableOutputs[1])
		local logCombinWeightsComplement = torch.log(1 - confidScores[1])--:expandAs(tableOutputs[1])
		local weightedExpert = nil
		local weightedSecondExpert = nil
		if direction == 1  then
			weightedExpert = torch.add(tableOutputs[1],logCombinWeights:expandAs(tableOutputs[1]))
			weightedSecondExpert = torch.add(tableOutputs[2],logCombinWeightsComplement:expandAs(tableOutputs[1]))
		elseif direction == 2 then
			weightedExpert = torch.add(tableOutputs[1],logCombinWeightsComplement:expandAs(tableOutputs[1]))
			weightedSecondExpert = torch.add(tableOutputs[2],logCombinWeights:expandAs(tableOutputs[1]))
		end
		local secondAdd  = torch.exp(weightedSecondExpert-weightedExpert)
		local oneTensor = torch.CudaTensor(secondAdd:size()):fill(1.0)
		secondAdd = torch.log(secondAdd+oneTensor)
		return weightedExpert + secondAdd
	end
	return comb
end

function logSumExpTable2(t)
	local secondAdd = torch.exp(t[2]-t[1])
	local oneTensor = torch.CudaTensor(secondAdd:size()):fill(1.0)
	secondAdd = torch.log(secondAdd+oneTensor)
	return t[1] + secondAdd
end
function confidenceMutualInterp(direction)
	local sys1 = direction 
	local sys2 = 2
	if direction == 2 then
		sys2 = 1
	end
	local k = 100	
	function comb(tableOutputs,confidScores)
		--print(confidScores[1][1])
		local logCombinWeights = torch.log(confidScores[1])--:expandAs(tableOutputs[1])
		local logCombinWeightsComplement = torch.log(1 - confidScores[1])--:expandAs(tableOutputs[1])
		
		local wExp1 = torch.add(tableOutputs[sys1],torch.log(confidScores[sys1]):expandAs(tableOutputs[sys1]))	
		local w2 = torch.cmul(confidScores[sys2],1-confidScores[sys1])
		local wExp2 = torch.add(tableOutputs[sys2],torch.log(w2):expandAs(tableOutputs[sys2]))
		local weightedExperts = logSumExpTable2({wExp1,wExp2})
		local wUnif = 1 - confidScores[sys1] - confidScores[sys2] - torch.cmul(confidScores[sys1],confidScores[sys2])
		local unifExp = 1/k * wUnif
		local unifEnsDistr = logSumExpTable2({torch.add(tableOutputs[1],torch.log(0.5)),torch.add(tableOutputs[2],torch.log(0.5))})		
		--local _,ind = weightedExperts:topk(k,true)
		local _,ind =unifEnsDistr:topk(k,true)
		for i=1,ind:size(1) do
			for j=1,k do
				weightedExperts[i][ind[i][j]] = torch.log(torch.exp(weightedExperts[i][ind[i][j]])+unifExp[i][1])
			end		
		end
		return weightedExperts 	
	end
	return comb
end


function confidenceMixture(confidenceScoreCombination)
	--local combination = 'arithmAve' 
	local combination = 'arithmAve' 

	function computeCombinationWeights(confidScores)
		if combination == 'arithmAve' then
			local norm = _.reduce(confidScores,function(memo,v) return memo+v end)
			norm = torch.log(norm)
			local logNormWeights = _.map(confidScores[1],function(i,v) return torch.log(v)-norm end)
			return logNormWeights
		elseif combination == 'softmax' then
			local norm = logSumExp2(torch.cat(confidScores))
			local logNormWeights = _.map(confidScores[1],function(i,v) return v - norm end)
			return logNormWeights
		end
	end	
			
	function comb(tableOutputs,confidScores)
		--[[print(confidScores[1])
		print(confidScores[2])
		print('---')]]---
		local logCombinWeights = computeCombinationWeights(confidScores)
		local weigtedExperts = _.map(tableOutputs, function(i,v) return v:add(logCombinWeights[i]:expandAs(v)) end)
		local secondAdd  = torch.exp(weigtedExperts[2]-weigtedExperts[1])
		local oneTensor = torch.CudaTensor(secondAdd:size()):fill(1.0)
		secondAdd = torch.log(secondAdd+oneTensor)
		return weigtedExperts[1] + secondAdd
	end
	return comb
end

function oracleSmoothing(weights,classes,maxK)
	local weights = weights
	local logWeights = _.map(weights,function(i,v) return torch.log(v) end)
	local classes = classes
	local maxK = maxK
	function weightedSum(tableOutputs) 
		local weigtedExperts = _.map(tableOutputs, function(i,v) return v:add(logWeights[i]) end)	
   		local secondAdd  = torch.exp(weigtedExperts[2]-weigtedExperts[1])
   		local oneTensor = torch.CudaTensor(secondAdd:size()):fill(1.0)
   		secondAdd = torch.log(secondAdd+oneTensor)
   		return weigtedExperts[1] + secondAdd
	end

	function comb(tableOutputs)
		-- weight it with 1 for now
		local _,expertCorrectIndex = tableOutputs[2]:topk(1,true)
		expertCorrectIndex = expertCorrectIndex[1]
		local _,rankedClasses = tableOutputs[1]:topk(classes[#classes],true)
		local rankOfCorrect = classes[#classes] + 1
		for i=1,rankedClasses:size(1) do
			if rankedClasses[i][1] == expertCorrectIndex then
				rankOfCorrect = i
			end
		end
		local smoothingDistr = nil
		if rankOfCorrect == classes[#classes] + 1 then
			smoothingDistr = uniformizeExpert_2(tableOutputs[1],maxK,classes[#classes])
		else
			local prevClass = 0
			for _,cl in ipairs(classes) do
				if rankOfCorrect > prevClass and rankOfCorrect < cl then
					smoothingDistr = uniformizeExpert_2(tableOutputs[1],cl,prevClass)
				else
					prevClass = cl
				end
			end
		end
		return weightedSum(tableOutputs[1],smoothingDistr)
	end
	return comb
end

		
		