-- author: Ke Tran <m.k.tran@uva.nl>
require 'tardis.GlimpseDot'
require 'optim'
require 'tardis.FastTransducer'
require 'moses'
require 'tardis.SeqAtt'
require 'nn.MSE'
local model_utils = require 'tardis.model_utils'

local utils = require 'misc.utils'

local confidenceCrossEntropy, parent = torch.class('nn.confidenceCrossEntropy', 'nn.Criterion')

function confidenceCrossEntropy:__init(weights, sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end

    self.output_tensor = torch.zeros(1)
    self.total_weight_tensor = torch.ones(1)
    self.target = torch.zeros(1):long()
end


function confidenceCrossEntropy:updateOutput(input, target)
   self.target = torch.CudaLongTensor and target:cudaLong() or target
   

   input.THNN.ClassNLLCriterion_updateOutput(
      input:cdata(),
      self.target:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage,
      THNN.optionalTensor(self.weights),
      self.total_weight_tensor:cdata()
   )
   self.output = self.output_tensor[1]
   return self.output, self.total_weight_tensor[1]
end

function confidenceCrossEntropy:updateGradInput(input, target)
   if type(target) == 'number' then
      if torch.typename(input):find('torch%.Cuda.*Tensor') then
          self.target = torch.CudaLongTensor and self.target:cudaLong() or self.target:cuda()
      else
          self.target = self.target:long()
      end
      self.target:resize(1)
      self.target[1] = target
   elseif torch.typename(input):find('torch%.Cuda.*Tensor') then
      self.target = torch.CudaLongTensor and target:cudaLong() or target
   else
      self.target = target:long()
   end

   self.gradInput:resizeAs(input):zero()

   input.THNN.ClassNLLCriterion_updateGradInput(
      input:cdata(),
      self.target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage,
      THNN.optionalTensor(self.weights),
      self.total_weight_tensor:cdata()
   )

   return self.gradInput
end

