-- This is NMT, you have to use CUDA
-- the code is written for GPUs
require 'nn'
require 'cutorch'
require 'cunn'

require 'data.loadBitex'
--require 'tardis.BiNMT' -- for the love of speed
require 'tardis.NMTA' -- for the love of speed
require 'tardis.BeamSearch'


local timer = torch.Timer()
torch.manualSeed(42)
local cfg = require 'pl.config'
local opt = cfg.read(arg[1])

if not opt.gpuid then opt.gpuid = 0 end
torch.manualSeed(opt.seed or 42)
cutorch.setDevice(opt.gpuid + 1)
cutorch.manualSeed(opt.seed or 42)

print('Experiment Setting: ', opt)
io:flush()


local loader = DataLoader(opt)
opt.padIdx = loader.padIdx

local model = nn.NMT(opt)

-- prepare data
function prepro(input)
    local x, y = unpack(input)
    local seqlen = y:size(2)
    -- make contiguous and transfer to gpu
    x = x:contiguous():cuda()
    prev_y = y:narrow(2, 1, seqlen-1):contiguous():cuda()
    next_y = y:narrow(2, 2, seqlen-1):contiguous():cuda()

    return x, prev_y, next_y
end

function train()
    local exp = math.exp
    local totwords = 0
    for epoch = 1, opt.maxEpoch do
        loader:train()
        model:training()
        local nll = 0
        local nbatches = loader.nbatches
        --print('number of batches: ', nbatches)
        for i = 1, nbatches do

            local x, prev_y, next_y = prepro(loader:next())
            --[[
            nll = nll + model:forward({x, prev_y}, next_y)
            model:backward({x, prev_y}, next_y)
            model:update(opt.learningRate)
            ]]
            nll = nll + model:optimize({x, prev_y}, next_y)
            model:clearState()
            totwords = totwords + prev_y:numel()
            if i % opt.reportEvery == 0 then
                print(string.format('epoch %d\t train ppl = %.4f speed = %.4f word/sec', epoch, exp(nll/i),  totwords / timer:time().real))
                xlua.progress(i, nbatches)
                collectgarbage()
            end
        end
        if epoch >= 3 then
            opt.learningRate = opt.learningRate * 0.5
        end

        timer:reset()
        -- not yet implemented
        loader:valid()
        model:evaluate()
        local valid_nll = 0
        local nbatches = loader.nbatches
        for i = 1, nbatches do
            local x, prev_y, next_y = prepro(loader:next())
            valid_nll = valid_nll + model:forward({x, prev_y}, next_y:view(-1))
            if i % 50 == 0 then collectgarbage() end
        end

        prev_valid_nll = valid_nll
        print(string.format('epoch %d\t valid perplexity = %.4f', epoch, exp(valid_nll/nbatches)))
        local checkpoint = string.format("%s/tardis_epoch_%d_%.4f.t7", opt.modelDir, epoch, valid_nll/nbatches)
        paths.mkdir(paths.dirname(checkpoint))
        print('save model to: ' .. checkpoint)
        --print('learningRate: ', opt.learningRate)
        model:save(checkpoint)

    end
end

local eval = opt.modelFile and opt.textFile

if not eval then
    -- training mode
    train()
else
    opt.transFile =  opt.transFile or 'translation.txt'

    local startTime = timer:time().real
    print('loading model...')
    model:load(opt.modelFile)
    local file = io.open(opt.transFile, 'w')
    -- create beam search object
    opt.vocab = loader.vocab
    local bs = BeamSearch(opt)
    bs:use(model)
    local refLine
    local nbLines = 0
    for line in io.lines(opt.textFile) do
        local translation = bs:run(line, opt.maxLength)
        print(translation)
        --nbLines = nbLines + 1
        --file:write(translation .. '\n')
        --file:flush()

    end
    file:close()
end
