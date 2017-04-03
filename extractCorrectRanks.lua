require 'translErrors.localTranslError'
require 'nn'
require 'cutorch'
require 'cunn'

require 'data.loadBitext'
require 'tardis.confidentNMT'
require 'tardis.SeqAtt'
local _ = require 'moses'

local cfg = require 'pl.config'
local opt = cfg.read(arg[1])
local ranksTableName = arg[2]


local loader = DataLoader(opt)
opt.padIdx = loader.padIdx
local targetVocIdx = loader:getTargetIdx() -- TODO: check if correct implementation
local metaSymbols = {} -- loader:getMetaSymbolsIdx()


function prepro(input)
    local x, y = unpack(input)
    local seqlen = y:size(2)
    -- make contiguous and transfer to gpu
    x = x:contiguous():cudaLong()
    prev_y = y:narrow(2, 1, seqlen-1):contiguous():cudaLong()
    next_y = y:narrow(2, 2, seqlen-1):contiguous():cudaLong()

    return x, prev_y, next_y
end




-- initialize model in evaluation mode
local model = nn.NMT(opt)
model:type('torch.CudaTensor')
--model:load(opt.confidenceModelToLoad)
model:load(opt.modelToLoad)

model:evaluate()

loader:loadForTesting(loader.tracker[1])
local nbatches = loader.nbatches


print("Number of batches: ",nbatches)
local ranksTable = {}
local confidenceTable = {}
for i = 1, nbatches do
	local x, prev_y, next_y = prepro(loader:next_origOrder())
    model:clearState()
    model:forward({x,prev_y},next_y)
    local correctRanks = model:extractRanksOfCorrect(next_y)
    table.insert(ranksTable,correctRanks)

end

torch.save(ranksTableName..'.t7',ranksTable)
