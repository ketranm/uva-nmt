-- Abstract class for Text processing
local AbstractDataLoader = torch.class('AbstractDataLoader')
local utf8 = require 'lua-utf8'
local _ = require 'moses'

function AbstractDataLoader:__init(startvocab)
    self._startvocab = startvocab or {'<pad>', '<s>', '</s>', '<unk>'}
    self._minfreq = 0
    self._maxsize = -1
    self.tracker = {{name = 'train', size = 0}, {name = 'valid', size = 0}}
end

function AbstractDataLoader:cutoff(threshold)
    -- map word with frequency less than threshold to '<unk>'
    self._minfreq = threshold
end

function AbstractDataLoader:shortlist(n)
    -- use shortlist, keep top n words
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


function AbstractDataLoader:loadForTesting(tracker)
    local fnames = {}
    for i = 1, tracker.fidx do
        local file = string.format('%s/%s.shard_%d.t7',
                                    self.dataPath, tracker.name, i)
        table.insert(fnames, file)
    end
    self._tensorfiles = fnames
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
    local unigrams = {}
    for i, w in ipairs(self._startvocab) do
        table.insert(idx2word, w)
        table.insert(unigrams, 1)
        word2idx[w] = #idx2word
    end

    for i = 1, vocabSize - #idx2word do
        local w = words[i]
        table.insert(idx2word, w)
        table.insert(unigrams, wordfreq[w])
        word2idx[w] = #idx2word
    end

    -- free memory
    collectgarbage()
    local vocab  = {word2idx = word2idx,
                    idx2word = idx2word,
                    word = function(idx) return idx2word[idx] or '<unk>' end,
                    idx = function(w) return word2idx[w] or word2idx['<unk>'] end,
                    size = #idx2word,
                    unigrams = unigrams}

    return vocab
end


function AbstractDataLoader.encodeString(s, vocab, padtype, vectorized)
    -- no reverse here
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

------------------------------------------------------
-- for Character Model
------------------------------------------------------
function AbstractDataLoader:buildchar(idx2word, maxlen)
    --[[ Map word to a tensor of character idx
    Parameters:
    - `idx2word`: contiguous table (no hole)
    - `maxlen`: truncate word if is length is excess this threshold
    Returns:
    - `word2char`: Tensor
    ]]
    -- compute max length of words
    local ll = 0 -- longest length
    for _, w in ipairs(idx2word) do
        ll = math.max(ll, utf8.len(w) + 2)
    end
    maxlen = math.min(ll, maxlen)
    print('max word length computed on the corpus: ' .. maxlen)
    -- special symbols
    local char2idx  = {['padl'] = 1, ['padr'] = 2}  -- padding
    local idx2char = {'padl', 'padr'}

    print('create char dictionary!')

    for _, w in ipairs(idx2word) do
        for _, c in utf8.next, w do
            if char2idx[c] == nil then
                idx2char[#idx2char + 1] = c
                char2idx[c] = #idx2char
            end
        end
    end

    local char = {idx2char = idx2char,
                  char2idx = char2idx,
                  padl = 1, padr = 2,
                  idx = function(c) return char2idx[c] end,
                  char = function(i) return idx2char[i] end}

    -- map words to tensor
    char.numberize = function(word, out)
                local x = {char.padl}
                for _, c in utf8.next, word do
                    x[#x + 1] = char.idx(c)
                end
                x[#x + 1] = char.padr
                local maxlen = out:numel()
                local x = torch.IntTensor(x)
                if x:numel() < maxlen then
                    out[{{1, x:numel()}}] = x
                else
                    out:copy(x[{{1, maxlen}}])
                end
                return out
            end

    local min = math.min
    local nwords = #idx2word
    -- we use zero for padding
    local word2char = torch.zeros(nwords, maxlen)
    for i, w in ipairs(idx2word) do
         char.numberize(w, word2char[i])
    end

    return word2char
end
