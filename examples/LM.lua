-- author: Ke Tran <m.k.tran@uva.nl>
require 'optim'
require 'cudnn'
local LM, parent = torch.class('nn.LM', 'nn.Module')

function LM:__init(opt)
    local V = opt.vocabSize
    local D = opt.inputSize
    local H = opt.hiddenSize
    local padidx = opt.padIdx
    self.net = nn.Sequential()
    self.net:add(nn.LookupTable(V, D, padidx, 1, 2))
    self.net:add(cudnn.GRU(D, H, opt.numLayers, true, opt.dropout))
    self.net:add(nn.Contiguous())
    self.net:add(nn.View(-1, H))
    self.net:add(nn.Linear(H, V))
    self.net:add(nn.LogSoftMax())

    local weights = torch.ones(V)
    weights[opt.padIdx] = 0
    self.criterion = nn.ClassNLLCriterion(weights, true)

    -- transfer to CUDA
    self.criterion:cuda()
    self.net:cuda()
    self.params, self.gradParams = self.net:getParameters()

    self.maxNorm = opt.maxNorm or 5
    -- for optim
    self.optimConfig = {}
    self.optimStates = {}
    self:reset(4e-2)
end

function LM:reset(std)
    self.params:uniform(-std, std)
end

function LM:forward(input, target)
    local target = target:view(-1)
    self.logp = self.net:forward(input)
    return self.criterion:forward(self.logp, target)
end

function LM:backward(input, target)
    -- zero grad manually here
    local dw = self.criterion:backward(self.logp, target:view(-1))
    self.net:backward(input, dw)
end

function LM:update(learningRate)
    local gradNorm = self.gradParams:norm()
    local scale = learningRate
    if gradNorm > self.maxNorm then
        scale = scale * self.maxNorm / gradNorm
    end
    self.params:add(self.gradParams:mul(-scale)) -- do it in-place
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
    self.net:training()
end

function LM:evaluate()
    self.net:evaluate()
end

function LM:load(fileName)
    local params = torch.load(fileName)
    self.params:copy(params)
end

function LM:save(fileName)
    torch.save(fileName, self.params)
end
