require 'cutorch'
require 'nn'
require 'cunn'
require 'tardis.EnsemblePrediction'
require 'data.loadMultiText'
local _ = require 'moses'
local cfg = require 'pl.config'
local timer = torch.Timer()
torch.manualSeed(42)
local cfg = require 'pl.config'
local opt = cfg.read(arg[1])
local multiOpt = {}
for c in opt.configFiles:gmatch("%S+") do
    local indivOpt =  cfg.read(c)
    indivOpt.padIdx = 1
    table.insert(multiOpt,indivOpt)
end
if not opt.gpuid then opt.gpuid = 0 end
torch.manualSeed(opt.seed or 42)
cutorch.setDevice(opt.gpuid + 1)
cutorch.manualSeed(opt.seed or 42)

print('Experiment Setting: ', opt)
io:flush()

function prepro(input)
    local xs, y = unpack(input)
    local seqlen = y:size(2)
    -- make contiguous and transfer to gpu
    xs = _.map(xs,function(i,x) return x:contiguous():cudaLong() end)
    prev_y = y:narrow(2, 1, seqlen-1):contiguous():cudaLong()
    next_y = y:narrow(2, 2, seqlen-1):contiguous():cudaLong()  --batchsize x seqlen
    return xs, prev_y, next_y
end

local ensemble = EnsemblePrediction(opt,multiOpt)
local loader = MultiDataLoader(opt,multiOpt)

local K = {5,20,100}--,1000}
local totalObservations = 0
local overlap = {}
local crossEntr = {}
local entropy = {}
for _,k in ipairs(K) do
    overlap[k] = 0
    crossEntr[k] = {0,0}
    entropy[k] = {0,0}
end

loader:train()
local nbatches = loader.nbatches
print('nbatches:: '..nbatches)
for i=1,nbatches do
    print('batch '..i)
    local xs, prev_y, next_y = prepro(loader:next_origOrder())
    local newOverlap, entr,crossE = ensemble:forwardAndComputeOutputOverlap(xs,prev_y,K)
    for k,cE in pairs(crossE) do
	overlap[k] = newOverlap[k] + overlap[k]
	crossEntr[k][1] = crossEntr[k][1] + cE[1]
	crossEntr[k][2] = crossEntr[k][2] + cE[2]
	entropy[k][1] = entropy[k][1] + entr[k][1]
	entropy[k][2] = entropy[k][2] + entr[k][2]
    end
    totalObservations = totalObservations + next_y:size(2)
end

for _,k in ipairs(K) do
    local normOverlap = overlap[k]/totalObservations
    local normKL_1 = symKLDiv[k][1]/totalObservations
    local normKL_2 = symKLDiv[k][2]/totalObservations

    print('topK:: '..k)
    print('class overlap::'..normOverlap)
    print('sym.KL::'..normKL_1 .. ' '..normKL_2)
end

