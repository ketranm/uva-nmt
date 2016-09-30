-- author: Ke Tran <m.k.tran@uva.nl>
require 'optim'
require 'cudnn'
require 'tardis.NCEModule'
require 'tardis.NCECriterion'
local model_utils = require 'tardis.model_utils'

local LM, parent = torch.class('nn.LM', 'nn.Module')

function LM:__init(opt)
    local V = opt.vocabSize
    local D = opt.inputSize
    local H = opt.hiddenSize
    local padidx = opt.padIdx
    self.net = nn.Sequential()
    self.net:add(nn.LookupTable(V, D, padidx, 1, 2))
    self.net:add(cudnn.LSTM(D, H, opt.numLayers, true, opt.dropout, false))
    self.net:add(nn.Contiguous())
    self.net:add(nn.View(-1, H))

    self.ncem = nn.NCEModule(H, V, 100, opt.unigrams, 1)
    self.ncec = nn.NCECriterion()
    self.criterion = nn.ClassNLLCriterion()

    -- transfer to CUDA
    self.criterion:cuda()
    self.net:cuda()
    self.ncem:cuda()
    self.ncec:cuda()

    self.params, self.gradParams =
            model_utils.combine_all_parameters(self.net, self.ncem)

    self.maxNorm = opt.maxNorm or 5
    -- for optim
    self.train = true
    self.optimConfig = {}
    self.optimStates = {}
end

function LM:reset(std)
    self.params:uniform(-std, std)
    self.ncem:reset()
end

function LM:forward(input, target)
    self.net:clearState()
    local target = target:view(-1)
    self.h = self.net:forward(input)
    if self.train == true then
        self.probs = self.ncem:forward{self.h, target}
        nll = self.ncec:forward(self.probs, target)
    else
        -- this is used for evaluation only
        local logp = self.ncem:forward({self.h, target})
        nll = self.criterion:forward(logp, target)
    end
    return nll
end

function LM:backward(input, target)
    local gradNCE = self.ncec:backward(self.probs, target:view(-1))
    local gradh = self.ncem:backward({self.h, target:view(-1)}, gradNCE)
    -- zero grad manually here
    self.net:backward(input, gradh[1])
end

function LM:learn(input, target, lr)
    local f = self:forward(input, target)
    self.gradParams:zero()
    self:backward(input, target)

    local gradNorm = self.gradParams:norm()
    local scale = lr
    if gradNorm > self.maxNorm then
        scale = scale * self.maxNorm / gradNorm
    end
    self.params:add(self.gradParams:mul(-scale)) -- do it in-place
    return f
end

function LM:optimize(input, target)
    local feval = function(x)
        if self.params ~= x then
            self.params:copy(x)
        end
        self.gradParams:zero()
        local f = self:forward(input, target)
        self:backward(input, target)
        local gradNorm = self.gradParams:norm()
        -- clip gradient
        if gradNorm > self.maxNorm then
            self.gradParams:mul(self.maxNorm / gradNorm)
        end
        return f, self.gradParams
    end
    local _, fx = optim.adam(feval, self.params, self.optimConfig, self.optimStates)
    return fx[1]
end

function LM:parameters()
    return self.params
end

function LM:training()
    self.train = true
    self.ncem.train = true
    self.ncem.normalized = false
    self.net:training()
end

function LM:evaluate()
    self.train = false
    self.ncem.train = false
    self.ncem.normalized = true
    self.ncem.logsoftmax = true
    self.net:evaluate()
end

function LM:load(fileName)
    local params = torch.load(fileName)
    self.params:copy(params)
end

function LM:save(fileName)
    torch.save(fileName, self.params)
end
