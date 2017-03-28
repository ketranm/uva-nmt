-- This is NMT, you have to use CUDA
-- the code is written for GPUs
require 'nn'
require 'cutorch'
require 'cunn'

require 'data.loadBitext'
require 'tardis.SeqAtt'
require 'tardis.BeamSearch'


local timer = torch.Timer()
local cfg = require 'pl.config'
local opt = cfg.read(arg[1])
if not opt.gpuid then opt.gpuid = 0 end
torch.manualSeed(opt.seed or 42)
cutorch.setDevice(opt.gpuid + 1)

print('Experiment Setting: ', opt)
io:flush()


local loader = DataLoader(opt)
opt.padIdx = loader.padIdx
require 'tardis.confidentNMT'
local model = nn.confidentNMT(opt)
model:type('torch.CudaTensor')

if opt.trainingScenario == 'confidenceMechanism' then
    local pretrainedModel = nn.NMT(opt)
    pretrainedModel:type('torch.CudaTensor')
    pretrainedModel:load(opt.modelToLoad)
    model:loadModelWithoutConfidence(pretrainedModel)
    print('t2')
    if opt.confidenceModelToLoad ~=nil then
	model:loadConfidence(opt.confidenceModelToLoad,opt)
	model.confidence:updateParameters(opt)
    else
    print('t3')
end
if opt.confidenceModelToLoad ~=nil then
	print('TT')
	model:loadConfidence(opt.confidenceModelToLoad)
	model.confidence:setLabelValue(opt)
end	
-- prepare data
function prepro(input)
    local x, y = unpack(input)
    local seqlen = y:size(2)
    -- make contiguous and transfer to gpu
    x = x:contiguous():cudaLong()
    prev_y = y:narrow(2, 1, seqlen-1):contiguous():cudaLong()
    next_y = y:narrow(2, 2, seqlen-1):contiguous():cudaLong()

    return x, prev_y, next_y
end

function train()
    local exp = math.exp
    local nupdates = 0
    for epoch = 1, opt.maxEpoch do
        loader:train()
        model:evaluate()
        model.confidence:training()
	if opt.trainingScenario == 'joint' then
		model:training()
	end
        local nll = 0
 	local confidMSE = {0,0}
	local aveLoss = 0 
        local nbatches = loader.nbatches
        local totwords = 0
        timer:reset()
        print('learningRate: ', opt.learningRate)
        print('number of batches', nbatches)
	model.confidence:clearStatistics()
        for i = 1,nbatches do
            collectgarbage()
            local x, prev_y, next_y = prepro(loader:next())
            model:clearState()
	    local new_nll,confidLoss = model:optimize({x,prev_y},next_y)
	    --local new_nll,confidLoss = model:forward({x,prev_y},next_y)
	    --model:backward({x,prev_y},next_y)
 	    --model:update(opt.learningRate)
	    local shareGood,shareBad = model:correctStatistics()
            nll = nll + new_nll
	    confidMSE[1] = confidMSE[1] + confidLoss[1]
	    confidMSE[2] = confidMSE[2] + confidLoss[2]
	    nupdates = nupdates + 1
	    totwords = totwords + prev_y:numel() 
            if i % opt.reportEvery == 0 then
		--model:changeObjectiveWeights()
                local floatEpoch = (i / nbatches) + epoch - 1
                local msg = 'epoch %.4f / %d   [ppl] %.4f [confidLoss] %.4f %.4f [speed] %.2f w/s [update] %.3f' --'epoch %.4f / %d   [ppl] %.4f [conf_mse] %.4f [ave_loss] %.4f  [speed] %.2f w/s [update] %.3f'
                --local args = {msg, floatEpoch, opt.maxEpoch, exp(nll/i),confidMSE/i,aveLoss/i, totwords / timer:time().real, nupdates/1000}
                local args = {msg, floatEpoch, opt.maxEpoch, exp(nll/i), confidMSE[1]/i, confidMSE[2]/i, totwords / timer:time().real, nupdates/1000}
                print(string.format(unpack(args)))
	    print('correct label stats: OK '..shareGood..' BAD '..shareBad)
            end
        end
        if epoch >= opt.decayAfter then
            opt.learningRate = opt.learningRate * 0.5
        end

        loader:valid()
        model:evaluate()
        model.confidence:evaluate()
        local mse = 0 -- validation loss
	local nll = 0
        local nbatches = loader.nbatches
        for i = 1, nbatches do
            model:clearState()
            local x, prev_y, next_y = prepro(loader:next())
	    local new_nll,confidLoss = model:forward({x, prev_y}, next_y:view(-1))
            mse = mse + confidLoss[1] 
	    nll = nll + new_nll
            mse = mse + confidLoss 
            if i % 50 == 0 then collectgarbage() end
        end

        
        local modelFile = ''
	if opt.trainingScenario == 'confidenceMechanism' then
		modelFile = string.format("%s_confid_ep_%d_%.4f.t7", opt.modelToLoad, epoch, confidMSE[1]/nbatches)
        else
 		string.format("%s/tardis_%d_%.4f.t7", opt.modelDir, epoch, nll/nbatches)
        end
        paths.mkdir(paths.dirname(modelFile))
        model:save(modelFile)

        local msg = '\nvalidation\nEpoch %d valid ppl %.4f valid mse %.4f\nsaved model %s'
        local args = {msg,epoch, exp(nll/nbatches),mse/nbatches, modelFile}
        print(string.format(unpack(args)))
    end
end

local eval = opt.modelFile and opt.textFile
if not eval then
    train()
else
    opt.transFile =  opt.transFile or 'translation.txt'

    local startTime = timer:time().real
    print(string.format('Loading model: %s', opt.modelFile))
    --model:load(opt.modelFile)
    local file = io.open(opt.transFile, 'w')
    -- create beam search object
    opt.vocab = loader.vocab
    local bs = BeamSearch(opt)
    bs:use(model)
    local refLine
    local nlines = 0
    for line in io.lines(opt.textFile) do
        local translation = bs:run(line, opt.maxLength)
        nlines = nlines + 1
        file:write(translation .. '\n')
        file:flush()
        io.write(string.format('translated sentence %d\r', nlines))
        io.flush()
    end
    file:close()
end
