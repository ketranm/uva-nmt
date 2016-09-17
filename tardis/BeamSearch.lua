-- beamsearch class
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
    print('special codes')
    print('bos', self.bosidx)
    print('eos', self.eosidx)
    print('unk', self.unkidx)
    print('pad', self.padidx)
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


function BeamSearch:run(x, maxLength)
    --[[
    Beam search:
    - x: source sentence
    - maxLength: maximum length of the translation
    - ref: opition, if it is provided, report sentence BLEU score
    ]]
    self.model:clearState()
    local K, Nw = self.K, 100 --self.Nw
    local x = encodeString(x, self.vocab[1].word2idx, self.reverseInput)
    local srcLength = x:size(2)
    local T = maxLength or utils.round(srcLength * 1.4)
    self.model:stepEncoder(x)

    local hypos = torch.CudaTensor(T, K):fill(self.bosidx)

    local curIdx = torch.CudaTensor(1, 1):fill(self.bosidx)
    local logProb = self.model:stepDecoder(curIdx)
    local maxScores, indices = logProb:topk(K, true)

    local scores = maxScores:view(-1, 1)
    hypos[1] = indices[1]
    self.model:repeatState(K)

    for t = 1, T-1 do
        local curIdx = hypos[t]:view(-1, 1)
        local logProb = self.model:stepDecoder(curIdx)
        local maxScores, indices = logProb:topk(Nw, true)
        local curScores = scores:repeatTensor(1, Nw)
        maxScores:add(curScores)

        local _scores, flatIdx = maxScores:view(-1):topk(K, true)
        scores = _scores:view(-1, 1)
        local nr, nc = maxScores:size(1), maxScores:size(2)

        local rowIdx = flatIdx:long():add(-1):div(nc):add(1):cuda()
        local colIdx = indices:view(-1):index(1, flatIdx)

        -- check stop
        local eos = colIdx:eq(self.eosidx)
        if t < 10 then
            print(hypos[{{1, t}, {}}])
        end
        if eos:sum() == 0 then
            hypos = hypos:index(2, rowIdx) -- next hypos
            hypos[t+1]:copy(colIdx)
            self.model:indexDecoderState(rowIdx)
        else
            print('found hypos at t = ', t)
        end

    end
    print('-----')
end
