require 'cutorch'
require 'nn'
require 'cunn'
require 'tardis.EnsemblePrediction'
require 'data.loadMultiText'
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
    local x, y = unpack(input)
    local seqlen = y:size(2)
    -- make contiguous and transfer to gpu
    x = x:contiguous():cudaLong()
    prev_y = y:narrow(2, 1, seqlen-1):contiguous():cudaLong()
    next_y = y:narrow(2, 2, seqlen-1):contiguous():cudaLong()  --batchsize x seqlen
    return x, prev_y, next_y
end

local ensemble = EnsemblePrediction(opt,multiOpt)
local loader = MultiDataLoader(opt)

local K = {5,20,100,1000}
local totalObservations = 0
local overlap = {}
local symKLDiv = {}
for _,k in ipairs(K) do
    overlap[k] = 0
    symKLDiv = {0,0}
end


local nbatches = loader.nbatches
for i=1,nbatches do
    local xs, prev_y, next_y = prepro(loader:next())
    local newOverlap,KL = ensemble:forwardAndComputeOutputOverlap(xs,prev_y,K)
    overlap = _.map(newOverlap,function(k,v) return overlap[k] + v end)
    symKLDiv = _.map(KL,function(k,v) return {symKLDiv[k][1] + v[1],symKLDiv[k][2]+v[2]} end)
    totalObservations = totalObservations + next_y:size(2)
end

for _,k in ipairs(L) do
    local normOverlap = overlap[k]/totalObservations
    local normKL = symKLDiv[k]/totalObservations
    print('topK:: '..k)
    print('class overlap::'..normOverlap)
    print('sym.KL::'..normKL[1] .. ' '..normKL[2])
end

