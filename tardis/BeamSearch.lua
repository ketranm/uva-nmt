local BeamSearch = torch.class('BeamSearch')
local _ = require 'moses'
local utils = require 'misc.utils'

function BeamSearch:__init(opt)
    self.vocab = opt.vocab
    self.K = opt.beamSize or 10
    self.Nw = opt.Nw or self.K
    self.reverseInput = opt.reverseInput or true

    self.bosidx = self.vocab[2].word2idx['<s>']
    self.eosidx = self.vocab[2].word2idx['</s>']
    self.unkidx = self.vocab[2].word2idx['<unk>']
    self.padidx = self.vocab[2].word2idx['<pad>']
    
    self._ignore = {[self.bosidx] = true, [self.eosidx] = true}
    self.normLength = true --opt.normLength
end


function BeamSearch:use(model)
    self.model = model
    self.model:evaluate() -- no dropout during testing
end

-- helper
function encodeString(input, word2idx, reverse)
    local xs = stringx.split(input)
    if reverse then
        xs = _.reverse(xs)
    end
    local output = torch.CudaTensor(#xs)
    for i, w in ipairs(xs) do
        output[i] = word2idx[w] or word2idx['<unk>']
    end
    return output:view(1, -1)
end

function decodeString(input, idx2word, ignore)
    -- from tensor to string
    local input = input:view(-1)
    local output = {}
    for i = 1, input:numel() do
        local idx = input[i]
        if  not ignore[idx] then
            table.insert(output, idx2word[idx])
        end
    end
    return table.concat(output, ' ')
end

function BeamSearch:run(x, maxLength)
    --[[
    Beam search:
    - x: source sentence
    - maxLength: maximum length of the translation
    - ref: opition, if it is provided, report sentence BLEU score
    ]]
    self.model:clearState()
    local K, Nw = self.K, self.Nw
    local x = encodeString(x, self.vocab[1].word2idx, self.reverseInput)
    local srcLength = x:size(2)
    local T = maxLength or utils.round(srcLength * 1.4)
    self.model:stepEncoder(x)

    local hypos = torch.CudaTensor(K, T):fill(self.bosidx)
    local curIdx = torch.CudaTensor(1, 1):fill(self.bosidx)
    local logProb = self.model:stepDecoder(curIdx)
    local maxScores, indices = logProb:topk(K, true)
    local scores = maxScores:view(-1, 1)

    hypos[{{}, 1}] = indices[1]
    self.model:repeatState(K)

    local nbestCands = {}
    local nbestScores = {}

    for t = 1, T-1 do
        local curIdx = hypos[{{}, {t}}]
        local logProb = self.model:stepDecoder(curIdx)
        local maxScores, indices = logProb:topk(Nw, true)
        local curScores = scores:repeatTensor(1, Nw)
        maxScores:add(curScores)

        local _scores, flatIdx = maxScores:view(-1):topk(K, true)
        scores = _scores:view(-1, 1)
        local nr, nc = maxScores:size(1), maxScores:size(2)

        local rowIdx = flatIdx:long():add(-1):div(nc):add(1):cuda()
        local colIdx = indices:view(-1):index(1, flatIdx)

        local xc = colIdx:eq(self.eosidx) --  completed candidates
        local nx = xc:sum()
        if nx > 0 then
            -- fond a candiate
            local cands = rowIdx:maskedSelect(xc)
            local completedHypos = hypos:index(1, cands):narrow(2, 1, t)
            local xscores = scores:index(1, xc):view(-1)
            for i = 1, nx do
                local output = decodeString(completedHypos[i], self.vocab[2].idx2word, self._ignore)
                local score = xscores[i]
                if self.normLength then
                    score = score / t
                end
                table.insert(nbestCands, output)
                table.insert(nbestScores, score)
            end
            print(nbestCands)
            if nx == colIdx:numel() then break end

            local rc = colIdx:ne(self.eosidx) -- remain hypotheses to expand
            rowIdx = rowIdx:maskedSelect(rc)
            colIdx = colIdx:maskedSelect(rc)
            scores = scores:maskedSelect(rc)
            K = rc:sum()
            if K == 0 then break end
        end

        hypos = hypos:index(1, rowIdx)
        hypos[{{}, {t+1}}] = colIdx
        self.model:indexDecoderState(rowIdx)
    end

    if #nbestCands == 0 then
        -- worst case scenario, we have to take the best possible candiate
        assert(K == self.K)
        scores = scores:view(-1)
        for i = 1, K do
            local output = decodeString(hypos[i], self.vocab[2].idx2word, self._ignore)
            local score = scores[i]
            table.insert(nbestCands, output)
            table.insert(nbestScores, score)
        end
    end

    -- pick the best translation candidates
    local score, idx = torch.Tensor(nbestScores):topk(1, true)
    return nbestCands[idx[1]]
end
