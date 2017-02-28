-- author: Ke Tran <m.k.tran@uva.nl>
require 'tardis.GlimpseDot'
require 'optim'
require 'tardis.FastTransducer'
require 'moses'
require 'tardis.SeqAtt'
require 'tardis.Confidence'
require 'tardis.EntropyConstraint'
local model_utils = require 'tardis.model_utils'

local utils = require 'misc.utils'

local NMT, parent = torch.class('nn.MaxEntNMT', 'nn.NMT')

function NMT:__init(opt)
    -- build encoder
    parent.__init(opt)
    
    self.entropyConstraint = nn.EntropyConstraint(opt)
    self.entropyConstraintWeight = opt.entropyConstraintWeight
    if self.entropyConstraintWeight == nil then
        self.entropyConstraintWeight = 1.0
    end
end

function NMT:forward(input,target)
    local ppl = parent.forward(input,target)
    local entropy = torch.mul(self.entropyConstraint(self.logProb),self.entropyConstraintWeight)
    return ppl,entropy
end

function NMT:backward(input,target)
    self.gradParams:zero()
    local gradXent = self.criterion:backward(self.logProb, target:view(-1))
    local gradEntropy = torch.mul(self.entropyConstraint:backward(self.logProb),self.entropyConstraintWeight))
    local gradLayer = self.layer:backward({self.cntx, self.decOutput}, gradXent+gradEntropy)
    local gradDecoder = gradLayer[2] -- grad to decoder
    local gradGlimpse =
        self.glimpse:backward({self.encOutput, self.decOutput}, gradLayer[1])

    gradDecoder:add(gradGlimpse[2]) -- accumulate gradient in-place

    self.decoder:backward(input[2], gradDecoder)

    -- initialize gradient from decoder
    local gradStates = self.decoder:gradStates()
    self.encoder:setGradStates(gradStates)
    -- backward to encoder
    local gradEncoder = gradGlimpse[1]
    self.encoder:backward(input[1], gradEncoder)
end

    



