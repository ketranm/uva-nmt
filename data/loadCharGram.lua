local DataLoader = torch.class('DataLoader')
local _ = require 'moses'
local utf8 = require 'lua-utf8'

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
    local w2cFile = path.join(opt.dataDir, 'word2chargram.t7')

    self.vocab = {}

    local trainPrefix = path.join(opt.dataDir, opt.trainPrefix)
    local validPrefix = path.join(opt.dataDir, opt.validPrefix)

    self.tracker = {train = {files = {}, nbatches = 0, prefix = trainPrefix},
                    valid = {files = {}, nbatches = 0, prefix = validPrefix}}

    if not path.exists(vocabFile) then
        print('run pre-processing, one-time setup!')
        print('creating source vocabulary ...')
        self.vocab[1] = self:_makeVocab(trainFiles[1], opt.sourceSize)

        print('creating word2char mapping for the source!')
        self.word2chargram, chargram2idx = self:_word2char(self.vocab[1].idx2word)
        print(self.word2char:size())
        torch.save(w2cFile, self.word2chargram)
        self.vocab.chargram2idx = chargram2idx
        print('creating target vocabulary ...')
        self.vocab[2] = self:_makeVocab(trainFiles[2], opt.targetSize)
        torch.save(vocabFile, self.vocab)

        print('create training tensor files...')
        self:text2tensor(trainFiles, opt.shardSize, opt.batchSize, self.tracker.train)

        print('create validation tensor files...')
        self:text2tensor(validFiles, opt.shardSize, opt.batchSize, self.tracker.valid)

        torch.save(indexFile, self.tracker)
    else
        self.word2char = torch.load(w2cFile)
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

function DataLoader:valid()
    -- no need shuffle for validation
    self._tensorfiles = _.shuffle(self.tracker.valid.files)
    self.isize = 0
    self.nbatches = self.tracker.valid.nbatches
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

    if vocabSize == -1 then
        -- use all
        vocabSize = #words
        print('vocab size:', vocabSize)
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
    local diff = 1 -- maximum different in length of the target
    local prime = 997 -- use a large prime number to avoid hash collision

    for source, target in seq.zip(unpack(files)) do
        nsents = nsents + 1

        local srcTokens = stringx.split(source, ' ')
        local srcLength = #srcTokens
        local trgTokens = stringx.split(target, ' ')
        local trgLength = #trgTokens + diff - (#trgTokens % diff)
        local trgLength = #trgTokens --+ diff - (#trgTokens % diff)
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
        local trgIdx = {trgVocab.word2idx['<s>']}
        for _, tok in ipairs(trgTokens) do
            local tokidx = trgVocab.word2idx[tok] or trgVocab.word2idx['<unk>']
            table.insert(trgIdx, tokidx)
        end
        table.insert(trgIdx, trgVocab.word2idx['</s>'])
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

function DataLoader:_word2chargram(idx2word)
    local bow = '<bow>' -- beginning of word
    local eow = '<eow>' -- end of word

    local function get_ngrams(input, n, output)
        local m = #input
        for i = 0, m-n do
            local cgx = {}
            for j = i+1, i+n do
                table.insert(cgx, input[j])
            end
            table.insert(output, table.concat(cgx, '-'))
        end
    end
    print('create chargram dictionary!')
    local chargram2idx = {}
    local idx2chargram = {}
    local wid2chargram = {}
    local maxnc = 0
    for _, w in ipairs(idx2word) do
        local chars = {bow}
        for _, char in utf8.next, w do
            table.insert(chars, char)
        end
        table.insert(chars, eow)
        local chargrams = {}
        for n = 2, order do
            get_ngrams(chars, n, chargrams)
        end

        local cgrams = {}
        for _, cgx in ipairs(chargrams) do
            if not chargram2idx[cgx] then
                table.insert(idx2chargram, cgx)
                chargram2idx[cgx] = #idx2chargram
            end
            table.insert(cgrams, chargram2idx[cgx])
        end
        table.insert(wid2chargram, cgrams)
        if maxnc < #cgrams then
            maxnc = #cgrams
        end
    end
    print('max number of chargrams / word:', maxnc)
    print('number of chargrams = ', #idx2chargram)
    local word2chargram = torch.ones(#idx2word, maxnc)
    for i, cgrams in iparis(wid2chargram) do
        for j = 1, #cgrams do
            word2chargram[i][j] = cgrams[j]
        end
    end
    return word2chargram, chargram2idx
end
