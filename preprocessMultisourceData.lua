require 'translErrors.localTranslError'
require 'nn'
require 'cutorch'
require 'cunn'

require 'data.loadMultiText'
require 'data.loadBitext'
require 'tardis.SeqAtt'
local _ = require 'moses'
local cfg = require 'pl.config'
local opt = cfg.read(arg[1])
local multiKwargs = _.map(_.slice(arg,2,#arg),function(i,v) return cfg.read(v) end)

--map multisource data to multisource tensors
local loader = MultiDataLoader(opt,multiKwargs)


-- save data in the same order for src-trg pairs independently
loader:loadForTesting(loader.tracker[1])
local sourceLoaders = _.map(multiKwargs,function(i,v) return DataLoader(v) end)
local nbatches = loader.nbatches
print("Number of batches: ",nbatches)
for i = 1, nbatches do
	local batch = loader:next_origOrder()
	local x = batch[1]
	local y = batch[2]
	for i=1,#sourceLoaders do
		sourceLoaders[i]:addBatch({x[i],y},multiKwargs[i].shardSize)
	end
    if i % opt.reportEvery == 0 then
    	print("Batch ",i)
    end
end
for i,l in ipairs(sourceLoaders) do
	local indexfile = multiKwargs[i].dataPath..'/index.t7' 
	l:saveCurrentShard(multiKwargs[i].shardSize,indexfile)
end


