

function computeUniformLog(topDistr)
	local denom = torch.Tensor(topDistr:size(1),1):fill(torch.log(topDistr:size(2)))
	local max,ind = topDistr:topk(1,true)
	max = max:expand(topDistr:size())
	local diff = topDistr - max:expand(topDistr:size())
	local logsumexp = max[{{},{1}}] + torch.log(torch.sum(torch.exp(diff),2))
	local logKUniform = logsumexp:csub(denom:cuda())
	return logKUniform
end

function totalLogMass(topDistr)
	local max,ind = topDistr:topk(1,true)
	max = max:expand(topDistr:size())
	local diff = topDistr - max:expand(topDistr:size())
	local logsumexp = max[{{},{1}}] + torch.log(torch.sum(torch.exp(diff),2))
	return logsumexp
end

function topKUniform(distribution,k)
	local topDistr,ind = distribution:topk(k,true)
	local uniformLog = computeUniformLog(topDistr)
	local result = distribution
	for i=1,ind:size(1) do
		for j=1,k do
			result[i][ind[i][j]] = uniformLog[i][1]
		end
	end
	return result,uniformLog
end


function topKUniform_2(distribution,k)
	local topDistr,ind = distribution:topk(k,true)
	local logValue = (-1)*torch.log(k)
	local result = torch.Tensor(distribution:size())
	for i=1,ind:size(1) do
		for j=1,k do
			result[i][ind[i][j]] = logValue 
		end
	end
	return result
end

function uniformizeExpert_1(logProb,upperTopK,maxK)
	-- keep top-k as it is, uniformize everything else
	local topDistr,ind = logProb:topk(upperTopK,true)
	local totalLogMassTopK = nil
	if upperTopK == 1 then
		totalLogMassTopK = topDistr
	else
		totalLogMassTopK = totalLogMass(topDistr)
	end
	local restNum = maxK - upperTopK 
	local unifValue = torch.log((1-torch.exp(totalLogMassTopK))/restNum)
	local result = torch.Tensor(logProb:size())
	for i=1,ind:size(1) do
		result[{i,{}}]:fill(unifValue[i][1])
		for j=1,ind:size(2) do
			result[i][ind[i][j]] = topDistr[i][j]
		end
	end
	return result:cuda()
end


function uniformizeExpert_2(logProb,upperTopK,lowerTopK)
	--uniformize within top-k, removing lowerTopK
	local beam = upperTopK - lowerTopK	
	local unifValue = torch.log((999+beam)/(1000*beam))
	local _,ind = logProb:topk(upperTopK,true)
	local result = torch.Tensor(logProb:size()):fill(torch.log(1/1000))
	 
	for i=1,ind:size(1) do			
		for j=lowerTopK+1,ind:size(2) do
			result[i][ind[i][j]] = unifValue 
		end
	end
	return result:cuda()
end




	
