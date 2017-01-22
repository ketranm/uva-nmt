require 'torch'
require 'nn'

require 'tardis.Ensemble'
require 'data.loadMultiText'
local cfg = require 'pl.config'
local opt = cfg.read(arg[1])

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


local ensemble = Ensemble:new(opt)

local transFilename =  kwargs.transFile or 'translation.txt'
local outfile = io.open(transFilename,"w")
local nbestFile = io.open(transFilename .. '.nbest', 'w')
local multiSrcLoader = MultiDataLoader:new(opt)

local nbLines = 0   
multiSrcLoader:loadTestFiles()
while true do
    nbLines = nbLines + 1
    local x_tuple = multiSrcLoader:nextSrcTuple()
    if x_tuple == nil then break end
    local translation,nbestList = ensemble:translate(x_tuple)
    outfile:write(translation..'\n')
    outfile:flush()
    nbestFile:write('SENTID=' .. nbLines .. '\n')
    nbestFile:write(table.concat(nbestList, '\n') .. '\n')
    nbestFile:flush()

end
    
io.close(nbestFile)
