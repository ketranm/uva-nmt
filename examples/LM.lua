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
    self.net:add(cudnn.LSTM(D, H, opt.numLayers, true, opt.dropout, false))
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
end

function LM:reset(std)
    self.params:uniform(-std, std)
end

function LM:forward(input, target)
    self.net:clearState()
    local target = target:view(-1)
    self.logp = self.net:forward(input)
    return self.criterion:forward(self.logp, target)
end

function LM:backward(input, target)
    -- zero grad manually here
    local dw = self.criterion:backward(self.logp, target:view(-1))
    self.net:backward(input, dw)
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
    self.net:training()
end

function LM:evaluate()
    self.net:evaluate()
end

function LM:sample(x, T, temperature)
    if not self.first then
        self.net:get(2).rememberState = true
        self.net:remove()
        self.first = true
    end
    self.net:clearState()
    local sampled = torch.Tensor(1, T):cuda()
    local T0 = x:size(2)
    sampled[{{}, {1, T0}}]:copy(x)
    -- rolling in
    self.net:forward(sampled[{{}, {1, T0-1}}])
    local scores = self.net:forward(sampled[{{}, {T0, T0}}])
    local first_t = T0 + 1
    for t = first_t, T do
        local probs = torch.div(scores, temperature):exp()
        probs:div(torch.sum(probs))
        local next_idx = torch.multinomial(probs, 1):view(1, 1)
        sampled[{{}, {t, t}}]:copy(next_idx)
        scores = self.net:forward(next_idx)
    end
    return sampled
end

function LM:load(fileName)
    local params = torch.load(fileName)
    self.params:copy(params)
end

function LM:save(fileName)
    torch.save(fileName, self.params)
end
