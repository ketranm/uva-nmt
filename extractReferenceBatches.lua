require 'data.loadBitext'

local cfg = require 'pl.config'
local opt = cfg.read(arg[1])

local loader = DataLoader(opt)
opt.padIdx = loader.padIdx
referenceFilePath = arg[2]

function prepro(input)
    local x, y = unpack(input)
    local seqlen = y:size(2)
    -- make contiguous and transfer to gpu
    x = x:contiguous():cudaLong()
    prev_y = y:narrow(2, 1, seqlen-1):contiguous():cudaLong()
    next_y = y:narrow(2, 2, seqlen-1):contiguous():cudaLong()

    return x, prev_y, next_y
end

local referenceCollection = {}
loader:train()
local nbatches = loader.nbatches
print("Number of batches: ",nbatches)
for i = 1, nbatches do
	local _, prev_y, next_y = prepro(loader:next_origOrder())
    table.insert(referenceCollection,{prev_y,next_y})
end

torch.save(referenceFilePath,errorCollection)
print("Saved to ",referenceFilePath)




