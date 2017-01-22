

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

function entropyConfidence()
	function negEntropy(distrib)
		local prob = torch.exp(distrib)
		return torch.sum(distrib:cmul(prob))
	end
	function softm(t)
		local e = torch.exp(vector)
		local Z = torch.sum(e)
		return e/Z
	end
	function comb(tableOutputs)
		local negEntropies = torch.Tensor(_.map(tableOutputs,function(i,v) return negEntropy(v) end))
		local weights = softm(negEntropies)
		local totalOut = torch.mul(tableOutputs[1],weights[1])
		for i=2,#tableOutputs do totalOut:add(torch.mul(tableOutputs[i],weights[i])) end
		return totalOut
	end

	return comb
end



