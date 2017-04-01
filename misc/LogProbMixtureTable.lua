local LogProbMixtureTable, parent = torch.class('nn.LogProbMixtureTable', 'nn.Module')

function LogProbMixtureTable:__init(dim)
   parent.__init(self)
   self.dim = dim
   self.size = torch.LongStorage() -- size of input (batchSize,gaterSize,1)
   self.batchSize = 0
   self.size2 = torch.LongStorage()
   self.backwardSetup = false
   self.gradInput = {} -- gradient passed back
end

-- gating weights are expected to be in log space!!
function LogProbMixtureTable:updateOutput(input) 
   local gaterInput, expertInputs = table.unpack(input)
   
   -- buffers 
   self._gaterView = self._gaterView or input[1].new() -- new() creates instance of the same type
   self._expert = self._expert or input[1].new()
   self._expertView = self._expertView or input[1].new()
   
   self.dimG = 2 -- dimention along which gaters vary?
   local batchSize = gaterInput:size(1)
   if gaterInput:dim() < 2 then
      self.dimG = 1
      self.dim = self.dim or 1
      batchSize = 1
   end
   self.dim = self.dim or 2
      
   
   if gaterInput:size(self.dimG) ~= #expertInputs then
      error"Should be one gater output per expert"
   end
   local expertInput = expertInputs[1]
   if self.batchSize ~= batchSize then
      self.size:resize(expertInput:dim()+1):fill(1) -- 
      if self.dimG > 1 then 
         self.size[1] = gaterInput:size(1)
      end
      self.size[self.dim] = gaterInput:size(self.dimG)
      self.output:resizeAs(expertInput) -- make the same size as 1st expert input
      self.backwardSetup = false
      self.batchSize = batchSize
   end
   self._gaterView:view(gaterInput, self.size) -- will be used to mix experts
   self.output:zero()

   -- multiply accumulate gater outputs by their commensurate expert
   local weigtedExperts = {}

   for i,expertInput in ipairs(expertInputs) do
      local gate = self._gaterView:select(self.dim,i):expandAs(expertInput)
      table.insert(weigtedExperts,expertInput:add(gate))
   end

   --[[local logProducts = torch.cat(combinedWeigtedExperts,3)
   local maxLogs,_ = logProducts:topk(1,true):expandAs(logProducts)
   local expSum = torch.Tensor(logProducts:size(1),logProducts:size(2))
   for i=1,logProducts:size(3) do
      expSum:add(torch.exp(logProducts[{},{},{i,i}]-maxLogs))
   end
   self.output = torch.log(expSum) + maxLogs

   return self.output
   ]]--
   local secondAdd  = torch.exp(weigtedExperts[2]-weigtedExperts[1])
   local oneTensor = torch.CudaTensor():expandAs(secondAdd):fill(1.0)
   secondAdd = torch.log(secondAdd+oneTensor)
   self.output = weigtedExperts[1] + secondAdd
   return self.output
end


 
function MixtureTable:updateGradInput(input, gradOutput)
   local gaterInput, expertInputs = table.unpack(input)
   nn.utils.recursiveResizeAs(self.gradInput, input)
   local gaterGradInput, expertGradInputs = table.unpack(self.gradInput)
   
   -- buffers
   self._sum = self._sum or input[1].new()
   self._expertView2 = self._expertView2 or input[1].new()
   self._expert2 = self._expert2 or input[1].new()
      

   if not self.backwardSetup then -- whether grad inputs have been initialized
      for i,expertInput in ipairs(expertInputs) do
         local expertGradInput = expertGradInputs[i] or expertInput:clone()
         expertGradInput:resizeAs(expertInput)
         expertGradInputs[i] = expertGradInput
      end
      gaterGradInput:resizeAs(gaterInput)
      self.backwardSetup = true
   end
   
   -- like CMulTable, but with broadcasting
   for i,expertGradInput in ipairs(expertGradInputs) do
      -- gater updateGradInput
      self._expert:cmul(gradOutput, torch.exp(self.output:mul(-1))) -- OLD: elementwise multiplication of grad from above and i-th expert values
      self._expert:cmul(torch.exp(expertGradInput))
      self._expert:mul(torch.exp(gaterInput[i]))
      
      if self.dimG == 1 then
         self._expertView:view(self._expert, -1)
      else
         self._expertView:view(self._expert, gradOutput:size(1), -1)
      end
      self._sum:sum(self._expertView, self.dimG)
      if self.dimG == 1 then
         gaterGradInput[i] = self._sum:select(self.dimG,1)
      else
         gaterGradInput:select(self.dimG,i):copy(self._sum:select(self.dimG,1))
      end

      gaterGradInput
      
      -- expert updateGradInput
      --local gate = self._gaterView:select(self.dim,i):expandAs(expertGradInput)
      --expertGradInput:cmul(gate, gradOutput)     
   end

   return self.gradInput
end

function LogProbMixtureTable:type(type, tensorCache)
   self._gaterView = nil
   self._expert = nil
   self._expertView = nil
   self._sum = nil
   self._expert2 = nil
   self._expertView2 = nil
   return parent.type(self, type, tensorCache)
end

function MLogProbixtureTable:clearState()
   nn.utils.clear(self, {
     '_gaterView',
     '_expert',
     '_expertView',
     '_sum',
     '_expert2',
     '_expertView2',
   })
   return parent.clearState(self)
end