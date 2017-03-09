
local EntropyConstraint,parent = torch.class('nn.EntropyConstraint', 'nn.Criterion')

function EntropyConstraint:__init(opt)
	if opt.normalizeEntropy == 1 then
		self.normalize = 1/torch.log(opt.targetSize)
	else
		self.normalize = 1
	end

end


function EntropyConstraint:forward(inputLogProb)
	local probDistr = torch.exp(inputLogProb)
	local negEntropy = torch.sum(torch.cmul(inputLogProb,probDistr),2)
	self.negEntropy = torch.mul(negEntropy,self.normalize)
	local averagedEntropy = torch.sum(self.negEntropy)/self.negEntropy:size(1)
	return averagedEntropy
end

function EntropyConstraint:backward(inputLogProb)
	local probDistr = torch.exp(inputLogProb)
	local normalizedLogProb = torch.mul(inputLogProb,self.normalize)
	torch.add(normalizedLogProb,self.negEntropy:expandAs(normalizedLogProb))
	local gradEntropy = torch.cmul(probDistr,normalizedLogProb)
	return gradEntropy
end


