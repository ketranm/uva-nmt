
local EntropyConstraint,parent = torch.class('nn.EntropyConstraint', 'nn.Criterion')

function EntropyConstraint:__init(opt)
	if opt.normalizeEntropy == 1 then
		self.normalize = 1/torch.log(opt.targetSize)
	else
		self.normalize = 1
	end

	if opt.approxEntropyK ~= nil  then
		self.approxEntropyK = opt.approxEntropyK
	else
		self.approxEntropyK = 0
	end

end

function getTopKDistribution(output,topK)
	
	local vals,_ = output:topk(topK,true)
	return vals
end

function getTopKDistributionNonZero(output,topK)
	local _,indices = output:topk(topK,true)
	local result = torch.zeros(output:size())
	for i=1,topK do
		result[indices[i]] = 1
		output[indices]
	end
	return result
end

function EntropyConstraint:forward(logProb)

	local inputLogProb = logProb:view(-1)
	local probDistr = torch.exp(inputLogProb)
	if self.approxEntropyK > 0 then
		probDistr = torch.cmul(probDistr,getTopKDistributionNonZero(inputLogProb,self.approxEntropyK))
	end
	local negEntropy = torch.sum(torch.cmul(inputLogProb,probDistr),2)
	self.negEntropy = torch.mul(negEntropy,self.normalize)
	local averagedEntropy = torch.sum(self.negEntropy)/self.negEntropy:size(1)
	return averagedEntropy
end

function EntropyConstraint:backward(inputLogProb)
	local probDistr = torch.exp(inputLogProb)
	if self.approxEntropyK > 0 then
		probDistr = torch.cmul(probDistr,getTopKDistributionNonZero(inputLogProb,self.approxEntropyK))
	end
	local normalizedLogProb = torch.mul(inputLogProb,self.normalize)
	torch.add(normalizedLogProb,self.negEntropy:expandAs(normalizedLogProb))
	local gradEntropy = torch.cmul(probDistr,normalizedLogProb)
	return gradEntropy
end


