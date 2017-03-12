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

	indivOpt =  cfg.read(c)
	indivOpt.padIdx = 1
	table.insert(multiOpt,indivOpt)
end
if not opt.gpuid then opt.gpuid = 0 end
torch.manualSeed(opt.seed or 42)
cutorch.setDevice(opt.gpuid + 1)
cutorch.manualSeed(opt.seed or 42)

print('Experiment Setting: ', opt)
io:flush()


local ensemble = EnsemblePrediction(opt,multiOpt)

local transFilename =  opt.transFile or 'translation.txt'
local outfile = io.open(transFilename,"w")
local multiSrcLoader = MultiDataLoader.newTestLoader(multiOpt)
local nlines = 0   
while true do
    nlines = nlines + 1
    local x_tuple = multiSrcLoader() --:nextSrcTuple()
    if x_tuple == nil then break end
    local translation = ensemble:translate(x_tuple)
    outfile:write(translation..'\n')
    outfile:flush()
    io.write(string.format('translated sentence %d\r', nlines))
    io.flush()
end
outfile:close()   
