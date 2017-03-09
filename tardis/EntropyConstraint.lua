
local EntropyConstraint,parent = torch.class('nn.EntropyConstraint', 'nn.Criterion')

function EntropyConstraint:__init(opt)
	if opt.normalizeEntropy == 1 then
		self.normalize = 1/torch.log(self.targetSize)
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
	
	local negEntropy = torch.sum(torch.cmul(inputLogProb,probDistr))
	self.entropy = torch.mul(negEntropy,-1*self.normalize)
	local averagedEntropy = torch.sum(self.entropy)/self.entropy:size(1)
	return averagedEntropy
endh

function EntropyConstraint:backward(inputLogProb)
	local probDistr = torch.exp(inputLogProb)
	if self.approxEntropyK > 0 then
		probDistr = torch.cmul(probDistr,getTopKDistributionNonZero(inputLogProb,self.approxEntropyK))
	end
	local normalizedLogProb = torch.mul(inputLogProb,self.normalize)
	normalizedLogProb:csub(self.entropy:expandAs(normalizedLogProb))
	local gradEntropy = torch.cmul(probDistr,normalizedLogProb)
	return gradEntropy
end


