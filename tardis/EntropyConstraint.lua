
local EntropyConstraint,parent = torch.class('nn.EntropyConstraint', 'nn.Criterion')

function EntropyConstraint:__init(opt)
	if opt.normalizeEntropy == 1 then
		self.normalize = 1/torch.log(self.targetSize)
	else
		self.normalize = 1
	end

end


function EntropyConstraint:forward(inputLogProb)
	local probDistr = torch.exp(inputLogProb)
	local negEntropy = torch.sum(torch.cmul(inputLogProb,probDistr))
	self.entropy = torch.mul(negEntropy,-1*self.normalize)
	local averagedEntropy = torch.sum(self.entropy)/self.entropy:size(1)
	return averagedEntropy
end

function EntropyConstraint:backward(inputLogProb)
	local probDistr = torch.exp(inputLogProb)
	local normalizedLogProb = torch.mul(inputLogProb,self.normalize)
	normalizedLogProb:csub(self.entropy:expandAs(normalizedLogProb))
	local gradEntropy = torch.sum(torch.cmul(probDistr,normalizedLogProb))
	return gradEntropy
end


