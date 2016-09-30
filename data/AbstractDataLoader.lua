-- Abstract class for Text processing
local AbstractDataLoader = torch.class('AbstractDataLoader')
local utf8 = require 'lua-utf8'
local _ = require 'moses'

function AbstractDataLoader:__init(startvocab)
    self._startvocab = startvocab or {'<pad>', '<s>', '</s>', '<unk>'}
    self._minfreq = 0
    self._maxsize = -1
    self.dataPath = ''
    self.tracker = {{name = 'train'}, {name = 'valid'}}
end

function AbstractDataLoader:cutoff(threshold)
    -- map word with frequency less than threshold to '<unk>'
    self._minfreq = threshold
end

function AbstractDataLoader:shortlist(n)
    self._maxsize = n
end

function AbstractDataLoader:load(tracker)
    local fnames = {}
    for i = 1, tracker.fidx do
        local file = string.format('%s/%s.shard_%d.t7',
                                    self.dataPath, tracker.name, i)
        table.insert(fnames, file)
    end
    self._tensorfiles = _.shuffle(fnames)
    self.isize = 0
    self.nbatches = tracker.size
end

function AbstractDataLoader:train()
    self:load(self.tracker[1])
end

function AbstractDataLoader:valid()
    self:load(self.tracker[2])
end

function AbstractDataLoader:next()
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


function AbstractDataLoader:tensorize()
    err('not yet implemented!')
end

function AbstractDataLoader:makeVocab(textfile)
    local wordfreq = {}
    print(string.format('reading in %s', textfile))
    for line in io.lines(textfile) do
        for w in line:gmatch('%S+') do
            wordfreq[w] = (wordfreq[w] or 0) + 1
        end
    end

    local words = {}
    for w in pairs(wordfreq) do
        words[#words + 1] = w
    end
    -- sort by frequency
    table.sort(words, function(w1, w2)
        return wordfreq[w1] > wordfreq[w2] or
            wordfreq[w1] == wordfreq[w2] and w1 < w2
    end)

    local vocabSize = 0
    if self._maxsize == -1 and self._minfreq == 0 then
        print('use the whole vocabulary!')
        vocabSize = #words + #self._startvocab
    elseif self._minfreq > 0 then
        for i, w in ipairs(words) do
            if wordfreq[w] < self._minfreq then
                vocabSize = i
                break
            end
        end
    else
        if #words < self._maxsize then
            print('shortlist > #types. Reset shortlist!')
            self._maxsize = #words
            vocabSize = self._maxsize
        else
            vocabSize = self._maxsize
        end
    end

    local word2idx = {}
    local idx2word = {}

    for i, w in ipairs(self._startvocab) do
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
    local vocab  = {word2idx = word2idx,
                    idx2word = idx2word,
                    word = function(idx) return idx2word[idx] or '<unk>' end,
                    idx = function(w) return word2idx[w] or 3 end,
                    size = #idx2word}

    return vocab
end


function AbstractDataLoader.encodeString(s, vocab, padtype, vectorized)
    local ws = stringx.split(s)
    local xs = {}
    local padtype = padtype or 'none'
    if padtype == 'first' or padtype == 'both' then
        xs[1] = vocab.idx('<s>')
    end

    for _, w in ipairs(ws) do
        table.insert(xs, vocab.idx(w))
    end

    if padtype == 'last' or padtype == 'both' then
        table.insert(xs, vocab.idx('</s>'))
    end

    if vectorized then
        return torch.IntTensor(xs)
    else
        return xs
    end
end

function AbstractDataLoader.decodeString(xs, vocab)
    local ws = {}
    local n = 0
    if torch.type(xs) == 'table' then
        n = #xs
    else
        n = xs:numel()
    end
    for i = 1, n do table.insert(ws, vocab.word(xs[i])) end
    return table.concat(ws, ' ')
end


function AbstractDataLoader:text2tensor(textfiles, shardSize, batchSize, tracker)
    error('not yet implemented!')
end

function AbstractDataLoader:characterize(idx2word, maxlen)
    --[[ Map word to a tensor of character idx
    Parameters:
    - `idx2word`: contiguous table (no hole)
    - `maxlen`: truncate word if is length is excess this threshold
    Returns:
    - `word2char`: Tensor
    ]]
    local bow = '<bow>' -- beginning of word
    local eow = '<eow>' -- end of word
    local pow = '<pow>' -- pad of word
    local char2idx  = {[pow] = 1, [bow] = 2, [eow] = 3}
    local idx2char = {pow, bow, eow}

    print('create char dictionary!')
    for _, w in ipairs(idx2word) do
        for _, char in utf8.next, w do
            if char2idx[char] == nil then
                idx2char[#idx2char + 1] = char
                char2idx[char] = #idx2char
            end
        end
    end

    local function w2chars(word)
        local chars = {char2idx[bow]}
        for _, char in utf8.next, word do
            chars[#chars + 1] = char2idx[char]
        end
        chars[#chars + 1] = char2idx[eow]

        return chars
    end

    local min = math.min
    local nwords = #idx2word

    local word2char = torch.ones(nwords, maxlen)
    for i, w in ipairs(idx2word) do
        local chars = w2chars(w)
        local v = word2char[i]
        for j = 1, min(#chars, maxlen) do
            v[j] = chars[j]
        end
    end
    return word2char
end
