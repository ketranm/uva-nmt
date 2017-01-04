require 'data.AbstractDataLoader'
local _ = require 'moses'

local DataLoader, parent = torch.class('DataLoader', 'AbstractDataLoader')

function DataLoader:__init(opt)
    parent.__init(self)
    -- data path
    self.dataPath = opt.dataPath
    self.langs = {opt.source, opt.target}

    local trainfiles = _.map(self.langs, function(i, ext)
        return path.join(self.dataPath, string.format('train.%s', ext) )
    end)
    local validfiles = _.map(self.langs, function(i, ext)
        return path.join(self.dataPath, string.format('valid.%s', ext) )
    end)

    -- helper
    local vocabfile = path.join(opt.dataPath, 'vocab.t7')
    -- auxiliary file to store additional information about shards
    local indexfile = path.join(opt.dataPath, 'index.t7')
    self.vocab = {}
    if not path.exists(vocabfile) then
        print('=> creating source vocabulary ...')
        self:shortlist(opt.sourceSize)
        self.vocab[1] = self.makeVocab(self, trainfiles[1])

        print('=> creating target vocabulary ...')
        self:shortlist(opt.targetSize)
        self.vocab[2] = self.makeVocab(self, trainfiles[2])
        torch.save(vocabfile, self.vocab)

        print('=> create training tensor files...')
        self:text2tensor(trainfiles, opt.shardSize, opt.batchSize, self.tracker[1])

        print('=> create validation tensor files...')
        self:text2tensor(validfiles, opt.shardSize, opt.batchSize, self.tracker[2])

        torch.save(indexfile, self.tracker)
    else
        self.vocab = torch.load(vocabfile)
        self.tracker = torch.load(indexfile)
    end
    self.padIdx = self.vocab[2].idx('<pad>')
    assert(self.padIdx == 1)
end



function DataLoader:buildCharSource(maxlen)
    return self:buildchar(self.vocab[1].idx2word, maxlen)
end

function DataLoader:saveShard(buckets, batchSize, tracker)
    local shard = {}
    for bidx, bucket in pairs(buckets) do
        -- make a big torch.IntTensor matrix
        local ss = torch.IntTensor(bucket[1]):split(batchSize, 1)
        local ts = torch.IntTensor(bucket[2]):split(batchSize, 1)
        buckets[bidx] = nil -- free memory
        -- sanity check
        assert(#ss == #ts)
        for i = 1, #ss do
            table.insert(shard, {ss[i], ts[i]})
        end
    end

    if not tracker.fidx then tracker.fidx = 0 end
    tracker.fidx = tracker.fidx + 1

    local file = string.format('%s/%s.shard_%d.t7',
                                self.dataPath, tracker.name, tracker.fidx)
    torch.save(file, shard)
    tracker.size = (tracker.size or 0) + #shard
    collectgarbage()
end

function DataLoader:text2tensor(textfiles, shardSize, batchSize, tracker)
    --[[Load source and target text file and save to tensor format.
    If the files are too large, process a shard of shardSize sentences at a time
    --]]
    local files = _.map(textfiles, function(i, file)
        return io.lines(file)
    end)
    -- helper
    local nsents = 0 -- sentence counter
    local buckets = {}
    local diff = 1 -- maximum different in length of the target
    local prime = 997 -- use a large prime number to avoid hash collision

    for source, target in seq.zip(unpack(files)) do
        nsents = nsents + 1
        local srcIdx = self.encodeString(source, self.vocab[1], 'none')
        local trgIdx = self.encodeString(target, self.vocab[2], 'both')
        local trgLen = #trgIdx + diff - (#trgIdx % diff)
        -- hashing
        local bidx =  prime * #srcIdx + trgLen
        -- reverse the source sentence
        srcIdx = _.reverse(srcIdx)

        for i = 1, trgLen - #trgIdx do
            table.insert(trgIdx, self.vocab[2].idx('<pad>'))
        end
        -- put sentence pairs to corresponding bucket
        buckets[bidx] = buckets[bidx] or {{}, {}}
        local bucket = buckets[bidx]
        table.insert(bucket[1], srcIdx)
        table.insert(bucket[2], trgIdx)

        if nsents % shardSize == 0 then
            self:saveShard(buckets, batchSize, tracker)
            buckets = {}
        end
    end

    if nsents % shardSize  > 1 then
        self:saveShard(buckets, batchSize, tracker)
    end
end
function DataLoader:getTargetIdx()
    local targetIdx = {}
    for i,w in pairs(self.vocab[2].idx2word) do
        table.insert(targetIdx,i)
    return targetIdx
end

function DataLoader:getMetaSymbolsIdx()
    local result = {}
    for _,s in ipairs(self._startvocab) do
        table.insert(result,self.vocab[2].idx(s))
    end
    return result
end
