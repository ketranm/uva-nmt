-- beamsearch class
local BeamSearch = torch.class('BeamSearch')

function BeamSearch:__init(opt)
    self.vocab = opt.vocab
    self.K = opt.beamSize or 10
    self.Nw = opt.Nw or self.K
    self.reverseInput = opt.reverseInput or true

    self.bosidx = self.vocab[2].idx2word['<s>']
    self.eosidx = self.vocab[2].idx2word['</s>']
    self._ignore = {[self.bosidx] = true, [self.eosidx] = true}
end


function BeamSearch:use(model)
    self.model = model
    self.model:evaluate() -- no dropout during testing
end

-- helper
local function encoderString(input, word2idx, reverse)
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

function BeamSearch:run(x, maxLength, ref)
    --[[
    Beam search:
    - x: source sentence
    - maxLength: maximum length of the translation
    - ref: opition, if it is provided, report sentence BLEU score
    ]]
    local K, Nw = self.K, self.Nw
    local x = encodeString(x, self.vocab[2].word2idx, self.reverseInput)
    local srcLength = x:size(2)
    local T = maxLength or utils.round(srcLength * 1.4)
    self.model:stepEncoder(x)
    local scores = torch.CudaTensor(K, 1):zero()
    local hypos = torch.CudaTensor(T, K):fill(self.bosidx)

    local curIdx = hypos[1]:view(-1, 1)
    local logProb = self.model:stepDecoder(curIdx)
    local maxScores, rowids, colids = logProb:topk(K, true)
    print(rowids)
    -- wait a minutes?
end
