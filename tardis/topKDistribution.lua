

function computeUniformLog(topDistr)
	local denom = torch.Tensor(topDistr:size(1),1):fill(torch.log(topDistr:size(2)))
	local max,ind = topDistr:topk(1,true)
	max = max:expand(topDistr:size())
	local diff = topDistr - max:expand(topDistr:size())
	local logsumexp = max[{{},{1}}] + torch.log(torch.sum(torch.exp(diff),2))
	local logKUniform = logsumexp:csub(denom:cuda())
	return logKUniform
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
