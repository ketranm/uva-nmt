-- This is NMT, you have to use CUDA
-- the code is written for GPUs
require 'nn'
require 'cutorch'
require 'cunn'
require 'data.loadText'
require 'examples.LM'
require 'xlua'

local cmd = torch.CmdLine()
cmd:text('GRU Language Model')
cmd:option('-datapath', '../data', 'location of data')
cmd:option('-modelDir', '../model', 'location of data')
cmd:option('-batchSize', 128, 'size of minibatch')
cmd:option('-inputSize', 1024, 'embedding size')
cmd:option('-hiddenSize', 1024, 'GRU size')
cmd:option('-dropout', 0.3 ,'dropout rate!')
cmd:option('-numLayers', 3, 'number of layers in stacked GRU')
cmd:option('-maxNorm', 5, 'max gradient norm')
cmd:option('-gpuid', 0, 'GPU device')
cmd:option('-maxEpoch', 100, 'max number of epochs')
cmd:option('-reportEvery', 50, 'print progress')
cmd:option('-lr', 1, 'learning rate')
cmd:option('-vocabSize', 50000, 'vocab size')
--cmd:option('-cutoff', 5, 'shortlist')
cmd:text()
opt = cmd:parse(arg or {})

local timer = torch.Timer()
torch.manualSeed(42)
if not opt.gpuid then opt.gpuid = 0 end
torch.manualSeed(opt.seed or 42)
cutorch.setDevice(opt.gpuid + 1)
cutorch.manualSeed(opt.seed or 42)

print('Experiment Setting: ', opt)
io:flush()
local loader = DataLoader(opt)
opt.padIdx = 1
opt.vocabSize = loader.vocabSize
print('vocabulary size', loader.vocabSize)

local model = nn.LM(opt)

-- prepare data
function prepro(x)
    local T = x:size(2)
    -- make contiguous and transfer to gpu
    local input = x:narrow(2, 1, T-1):contiguous():cuda()
    local target = x:narrow(2, 2, T-1):contiguous():cuda()
    return input, target
end

function eval()
    loader:valid()
    model:evaluate()
    local nll = 0 -- validation loss
    local nbatches = loader.nbatches
    for i = 1, nbatches do
        local x, y = prepro(loader:next())
        nll = nll + model:forward(x, y)
        if i % 200 == 0 then collectgarbage() end
    end
    return nll / nbatches
end

function train()
    local exp = math.exp
    local nupdates = 0
    for epoch = 1, opt.maxEpoch do
        loader:train()
        model:training()
        local nll = 0
        local nbatches = loader.nbatches
        local totwords = 0
        timer:reset()
        for i = 1, nbatches do
            local x, y = prepro(loader:next())
            nll = nll + model:optimize(x, y)
            --nll = nll + model:learn(x, y, opt.lr)
            nupdates = nupdates + 1
            totwords = totwords + y:numel()
            if i % opt.reportEvery == 0 then
                --xlua.progress(i, nbatches)
                local floatEpoch = (i / nbatches) + epoch - 1
                local msg = 'epoch %.4f / %d   [ppl] %.4f   [speed] %.2f w/s [update] %.3f'
                local args = {msg, floatEpoch, opt.maxEpoch, exp(nll/i), totwords / timer:time().real, nupdates/1000}
                print(string.format(unpack(args)))
                collectgarbage()
            end

            if nupdates % 3000 == 0 then
                opt.lr = opt.lr * 0.9
            end
            --[[
            if nupdates % 10000 == 0 then
                local modelFile = string.format('%s/checkpoint_%d.t7', opt.modelDir, nupdates)
                paths.mkdir(paths.dirname(modelfile))
                model:save(modelfile)
            end]]
        end

        local nll = eval()
        local modelFile = string.format("%s/tardis_%d_%.4f.t7", opt.modelDir, epoch, nll)
        paths.mkdir(paths.dirname(modelFile))
        model:save(modelFile)

        local msg = '\nvalidation\nEpoch %d valid ppl %.4f\nsaved model %s'
        local args = {msg, epoch, exp(nll), modelFile}
        print(string.format(unpack(args)))
    end
end

modelfile = '/var/scratch/mktran/lmtest/tardis_17_4.2338.t7'
model:load(modelfile)

local s = 'today is a'
local x = loader.encodeString(s, loader.vocab, 'first', true)
x = x:view(1, -1):cuda()
for i = 1, 50 do
    --model:clearState()
    out = model:sample(x, 50, 1)
    local s = loader.decodeString(out:view(-1), loader.vocab)
    print(s)
end

--train()
