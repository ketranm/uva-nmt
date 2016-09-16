-- This is NMT, you have to use CUDA
-- the code is written for GPUs
require 'nn'
require 'cutorch'
require 'cunn'

require 'data.loadBitex'
require 'tardis.BiNMT' -- for the love of speed
--require 'search.Beam'


local timer = torch.Timer()
torch.manualSeed(42)
local cfg = require 'pl.opt'
local opt = cfg.read(arg[1])

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
    local seqlen = target:size(2)
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
            nll = nll + model:optimize({x, prev_y}, next_y)
            model:clearState()
            totwords = totwords + prev_y:numel()
            if i % opt.reportEvery == 0 then
                xlua.progress(i, nbatches)
                print(string.format('epoch %d\t train ppl = %.4f speed = %.4f word/sec', epoch, exp(nll/i),  totwords / timer:time().real))
                collectgarbage()
            end
        end
        timer:reset()
        -- not yet implemented
        loader:readValid()
        model:evaluate()
        local valid_nll = 0
        local nbatches = loader:nbatches()
        for i = 1, nbatches do
            local src, trg, nextTrg = prepro(loader:nextBatch())
            valid_nll = valid_nll + model:forward({src, trg}, nextTrg:view(-1))
            if i % 50 == 0 then collectgarbage() end
        end

        prev_valid_nll = valid_nll
        print(string.format('epoch %d\t valid perplexity = %.4f', epoch, exp(valid_nll/nbatches)))
        local checkpoint = string.format("%s/tardis_epoch_%d_%.4f.t7", opt.modelDir, epoch, valid_nll/nbatches)
        paths.mkdir(paths.dirname(checkpoint))
        print('save model to: ' .. checkpoint)
        print('learningRate: ', opt.learningRate)
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
    local loadTime = timer:time().real - startTime
    print(string.format('done, loading time: %.4f sec', loadTime))
    timer:reset()

    local file = io.open(opt.transFile, 'w')
    local nbestFile = io.open(opt.transFile .. '.nbest', 'w')
    -- if reference is provided compute BLEU score of each n-best
    local refFile
    if opt.refFile then
        refFile = io.open(opt.refFile, 'r')
    end

    -- create beam search object
    opt.srcVocab, opt.trgVocab = unpack(loader.vocab)
    local bs = BeamSearch(opt)
    bs:use(model)

    local refLine
    local nbLines = 0
    for line in io.lines(opt.textFile) do
        nbLines = nbLines + 1
        if refFile then refLine = refFile:read() end
        local translation, nbestList = bs:search(line, opt.maxTrgLength, refLine)
        file:write(translation .. '\n')
        file:flush()
        if nbestList then
            nbestFile:write('SENTID=' .. nbLines .. '\n')
            nbestFile:write(table.concat(nbestList, '\n') .. '\n')
            nbestFile:flush()
        end
    end
    file:close()
    nbestFile:close()

    local transTime = timer:time().real
    print(string.format('Done (%d) sentences translated', nbLines))
    print(string.format('Total time: %.4f sec', transTime))
    print(string.format('Time per sentence: %.4f', transTime/nbLines))
end
