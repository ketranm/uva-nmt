local _ = require 'moses'

function scalarCombination(scWeights,outputDim,inputType)
	local weights = scWeights
	--print(weights)
	if inputType == 'prob' then
		local logWeights = _.map(weights,function(i,v) return torch.log(v) end)
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
function entropyConfidence()
	local temperature = 1
	local topK = 0 
	function getTopKDistributions(tableOutputs)
		local result = {}
		for _,d in ipairs(tableOutputs) do
			local vals,_ = d:topk(topK,true)
			table.insert(result,vals)
		end
		return result
	end

	function negEntropy(distrib)
		local const = torch.mul(distrib[{{},{1}}],-1)
		local modifDistrib = torch.add(distrib,const:expand(distrib:size())) --:expand(distrib:size())
		local modifProb = torch.exp(modifDistrib)
		local modifEnt = torch.mul(torch.sum(torch.cmul(modifDistrib,modifProb),2),-1)
		local secondAdd = torch.cmul(torch.sum(modifProb,2),const)
		local result = modifEnt + secondAdd
		return torch.mul(torch.cmul(result,torch.exp(torch.mul(const,-1))),-1) 
	end
	function comb(tableOutputs)
		collectgarbage()
		if topK > 0 then
			tableOutputs =  getTopKDistributions(tableOutputs)
		end
		local negEntropies = torch.cat(_.map(tableOutputs,function(i,v) return negEntropy(v) end))
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



