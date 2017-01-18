require 'translErrors.localTranslError'
require 'nn'
require 'cutorch'
require 'cunn'

require 'data.loadMultiText'
require 'data.loadBiext'
require 'tardis.SeqAtt'
local cfg = require 'pl.config'
local opt = cfg.read(arg[1])
local multiKwargs = _.map(_.slice(arg,2,#arg),function(i,v) cfg.read(v) end)

--map multisource data to multisource tensors
local loader = MultiDataLoader(opt,multiKwargs)


-- save data in the same order for src-trg pairs independently
loader:loadForTesting(self.tracker[1])
local sourceLoaders = _.map(multiKwargs,function(i,v) return DataLoader(v) end)
local nbatches = loader.nbatches
print("Number of batches: ",nbatches)
for i = 1, nbatches do
	local x, y = loader:next_origOrder()
	for i=1,#x do
		sourceLoaders[i]:addBatch({x[i],y},multiKwargs[i].shardSize)
	end
    if i % opt.reportEvery == 0 then
    	print("Batch ",i)
    end
end
for _,l in sourceLoaders do l:saveCurrentShard() end


