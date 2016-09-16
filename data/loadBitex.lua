local DataLoader = torch.class('DataLoader')
local _ = require 'moses'

function DataLoader:__init(opt)
    self._startVocab = {'<s>', '</s>', '<unk>', '<pad>'}

    local langs = {opt.src, opt.trg}

    -- data path
    local trainFiles = _.map(langs, function(i, ext)
        return path.join(opt.dataDir, string.format('%s.%s', opt.trainPrefix, ext))
    end)

    local validFiles = _.map(langs, function(i, ext)
        return path.join(opt.dataDir, string.format('%s.%s', opt.validPrefix, ext))
    end)

    -- helper
    local vocabFile = path.join(opt.dataDir, 'vocab.t7')
    -- auxiliary file to store additional information about shards
    local indexFile = path.join(opt.dataDir, 'index.t7')

    self.vocab = {}

    local trainPrefix = path.join(opt.dataDir, opt.trainPrefix)
    local validPrefix = path.join(opt.dataDir, opt.validPrefix)

    self.tracker = {train = {files = {}, nbatches = 0, prefix = trainPrefix},
                    valid = {files = {}, nbatches = 0, prefix = validPrefix}}

    if not path.exists(vocabFile) then
        print('run pre-processing, one-time setup!')
        print('creating source vocabulary ...')
        self.vocab[1] = self:_makeVocab(trainFiles[1], opt.srcVocabSize)

        print('creating target vocabulary ...')
        self.vocab[2] = self:_makeVocab(trainFiles[2], opt.trgVocabSize)
        torch.save(vocabFile, self.vocab)

        print('create training tensor files...')
        self:text2tensor(trainFiles, opt.shardSize, opt.batchSize, self.tracker.train)

        print('create validation tensor files...')
        self:text2tensor(validFiles, opt.shardSize, opt.batchSize, self.tracker.valid)

        torch.save(indexFile, self.tracker)
    else
        self.vocab = torch.load(vocabFile)
        self.tracker = torch.load(indexFile)
    end
    self.padIdx = self.vocab[2].word2idx['<pad>']

end

function DataLoader:train()
    self._tensorfiles = _.shuffle(self.tracker.train.files)
    self.isize = 0
    self.nbatches = self.tracker.train.nbatches
end

function DataLoader:next()
    local idx = 0
    if self.isize == 0 then
        local file = table.remove(self._tensorfiles)
        self.data = torch.load(file)
        self.isize = #self.data
        self.permidx = torch.randperm(self.isize)
        idx = self.permidx[self.isize]
    else
        idx = self.permidx[self.isize]
    end
    self.isize = self.isize - 1
    return self.data[idx]
end

function DataLoader:_makeVocab(textFile, vocabSize)
    --[[ Create vocabulary with maximum vocabSize words.
    Parameters:
    - `textFile` : source or target file, tokenized, lowercased
    - `vocabSize` : the number of top frequent words in the textFile
    --]]

    local wordFreq = {}
    print('reading in ' .. textFile)
    for line in io.lines(textFile) do
        for w in line:gmatch('%S+') do
            wordFreq[w] = (wordFreq[w] or 0) + 1
        end
    end

    local words = {}
    for w in pairs(wordFreq) do
        words[#words + 1] = w
    end
    -- sort by frequency
    table.sort(words, function(w1, w2)
        return wordFreq[w1] > wordFreq[w2] or
            wordFreq[w1] == wordFreq[w2] and w1 < w2
    end)

    local word2idx = {}
    local idx2word = {}
    for i, w in ipairs(self._startVocab) do
        table.insert(idx2word, w)
        word2idx[w] = #idx2word
    end

    for i = 1, vocabSize - #idx2word do
        local w = words[i]
        table.insert(idx2word, w)
        word2idx[w] = #idx2word
    end
    -- free memory
    collectgarbage()
    return {word2idx = word2idx, idx2word = idx2word}
end

function DataLoader:_createShard(buckets, batchSize, tracker)
    local shard = {}
    for bidx, bucket in pairs(buckets) do
        -- make a big torch.IntTensor matrix
        local srcData = torch.IntTensor(bucket[1]):split(batchSize, 1)
        local trgData = torch.IntTensor(bucket[2]):split(batchSize, 1)
        buckets[bidx] = nil -- free memory
        -- sanity check
        assert(#srcData == #trgData)
        for i = 1, #srcData do
            table.insert(shard, {srcData[i], trgData[i]})
        end
    end

    local idx = #tracker.files + 1
    local file = string.format('%s.shard_%d.t7', tracker.prefix, idx)
    table.insert(tracker.files, file)
    torch.save(file, shard)
    tracker.nbatches = tracker.nbatches + #shard
end

function DataLoader:text2tensor(textFiles, shardSize, batchSize, tracker)
    --[[Load source and target text file and save to tensor format.
    If the files are too large, process a shard of shardSize sentences at a time
    --]]

    local files = _.map(textFiles, function(i, file)
        return io.lines(file)
    end)

    local srcVocab, trgVocab = unpack(self.vocab)
    -- helper
    local nsents = 0 -- sentence counter
    local buckets = {}
    local diff = 10 -- maximum different in length of the target
    local prime = 107

    for source, target in seq.zip(unpack(files)) do
        nsents = nsents + 1

        local srcTokens = stringx.split(source, ' ')
        local srcLength = #srcTokens
        local trgTokens = stringx.split(target, ' ')
        local trgLength = #trgTokens + diff - (#trgTokens % diff)
        -- hashing
        local bidx =  prime * srcLength + trgLength
        -- reverse the source sentence
        local srcIdx = {}
        for i = #srcTokens, 1, -1 do
            local tok = srcTokens[i]
            local tokidx = srcVocab.word2idx[tok] or srcVocab.word2idx['<unk>']
            table.insert(srcIdx, tokidx)
        end

        -- add BOS and EOS to target
        local trgIdx = {trgVocab['<s>']}
        for _, tok in ipairs(trgTokens) do
            local tokidx = trgVocab.word2idx[tok] or trgVocab.word2idx['<unk>']
            table.insert(trgIdx, tokidx)
        end
        table.insert(trgIdx, trgVocab['</s>'])
        -- add PAD to the end after EOS
        for i = 1, trgLength - #trgTokens do
            table.insert(trgIdx, trgVocab.word2idx['<pad>'])
        end

        -- put sentence pairs to corresponding bucket
        buckets[bidx] = buckets[bidx] or {{}, {}}
        local bucket = buckets[bidx]
        table.insert(bucket[1], srcIdx)
        table.insert(bucket[2], trgIdx)

        if nsents % shardSize == 0 then
            self:_createShard(buckets, batchSize, tracker)
            buckets = {}
        end
    end

    if nsents % shardSize  > 1 then
        self:_createShard(buckets, batchSize, tracker)
    end
end
