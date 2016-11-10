require 'data.AbstractDataLoader'
local DataLoader, parent = torch.class('DataLoader', 'AbstractDataLoader')
function DataLoader:__init(opt)
    parent.__init(self)
    -- data path
    self.dataPath = opt.dataPath
    local trainfile = path.join(self.dataPath, 'train.txt')
    local validfile = path.join(self.dataPath, 'valid.txt')
    local vocabfile = path.join(self.dataPath, 'vocab.t7')
    local indexfile = path.join(self.dataPath, 'index.t7')

    if opt.vocabSize then
        self:shortlist(opt.vocabSize)
    elseif opt.cutoff then
        self:cutoff(opt.cutoff)
    end

    self.tracker = {{name = 'train'}, {name = 'valid'}}
    local shardSize = opt.shardSize or 500000
    local batchSize = opt.batchSize or 128
    if not path.exists(vocabfile) then
        print('run pre-processing, one-time setup!')
        print('creating source vocabulary ...')
        self.vocab = self:makeVocab(trainfile)
        torch.save(vocabfile, self.vocab)

        print('create training tensor files...')
        self:tensorize(trainfile, shardSize, batchSize, self.tracker[1])

        print('create validation tensor files...')
        self:tensorize(validfile, shardSize, batchSize, self.tracker[2])
        torch.save(indexfile, self.tracker)
    else
        self.vocab = torch.load(vocabfile)
        self.tracker = torch.load(indexfile)
    end
    self.vocabSize = self.vocab.size
end


function DataLoader:word2char(maxlen)
    return self:buildchar(self.vocab.idx2word, maxlen or 15)
end

function DataLoader:saveShard(buckets, batchSize, tracker)
    local shard = {}
    for bx, bucket in pairs(buckets) do
        local xs = torch.IntTensor(bucket):split(batchSize, 1)
        bucket[bx] = nil -- free memeory
        for _, x in ipairs(xs) do
            table.insert(shard, x)
        end
    end
    if not tracker.fidx then tracker.fidx = 0 end
    tracker.fidx = tracker.fidx + 1

    local file = string.format('%s/%s.shard_%d.t7',
                                self.dataPath, tracker.name, tracker.fidx)
    torch.save(file, shard)
    tracker.size = tracker.size or 0 + #shard
    collectgarbage()
end

function DataLoader:tensorize(textfile, shardSize, batchSize, tracker)
    --[[Load source and target text file and save to tensor format.
    If the files are too large, process a shard of shardSize sentences at a time
    --]]

    local nsents = 0 -- sentence counter
    local buckets = {}

    for line in io.lines(textfile) do
        nsents = nsents + 1
        local xs = self.encodeString(line, self.vocab, 'both')
        local bx = #xs + 5 - (#xs % 5)
        if not buckets[bx] then buckets[bx] = {} end
        -- padding
        for i = 1, bx - #xs do
            table.insert(xs, self.vocab.idx('<pad>'))
        end
        table.insert(buckets[bx], xs)
        if nsents % shardSize == 0 then
            self:saveShard(buckets, batchSize, tracker)
            buckets = {}
        end

    end

    if nsents % shardSize > 0 then
        self:saveShard(buckets, batchSize, tracker)
    end
end
