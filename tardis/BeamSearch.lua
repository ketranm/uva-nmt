local BeamSearch = torch.class('BeamSearch')
local _ = require 'moses'
local utils = require 'misc.utils'

function BeamSearch:__init(opt)
    self.vocab = opt.vocab
    self.K = opt.beamSize or 10
    self.Nw = opt.Nw or self.K * self.K
    self.reverseInput = opt.reverseInput or true
    self.normLength = opt.normLength or 0 -- 1 if we normalized scores by length

    self.bosidx = self.vocab[2].word2idx['<s>']
    self.eosidx = self.vocab[2].word2idx['</s>']
    self.unkidx = self.vocab[2].word2idx['<unk>']
    self.padidx = self.vocab[2].word2idx['<pad>']

    self._ignore = {[self.bosidx] = true, [self.eosidx] = true}
    self.normLength = opt.normLength or 1
    
    self.aggrEntr = 0
    self.numPred = 0
end

function BeamSearch:getAveEnt()
	local result = self.aggrEntr/self.numPred
	return result
end

function BeamSearch:use(model)
    self.model = model
    self.model:evaluate() -- no dropout during testing
end

-- helper
function encodeString(input, vocab, reverse)
    local xs = stringx.split(input)
    if reverse then
        xs = _.reverse(xs)
    end
    local output = torch.CudaTensor(#xs)
    for i, w in ipairs(xs) do
        output[i] = vocab.idx(w)
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

function prune(maxscores,indices)
    local currBeam = maxscores:size(1)
    local threshold = 2 
    local pruneFactor = 1/2
    local pruneFrom = maxscores:size(2)*pruneFactor 
    local newmaxscores = torch

    for i=1,currBeam do
	local ent = entropy(maxscores[i])
	if ent  < threshold then
		for j=pruneFrom,maxscores:size(2) do
			maxscores[i][j] = -math.huge
		end
	end
   end	
   return maxscores,indices		
		 
	
end

function entropy(distrib)
	--local result = 0
	--for i=1,distrib:size(1) do
	    local prob = torch.exp(distrib)
      	    local result = -1 * torch.sum(torch.cmul(distrib,prob))

	return result
    end
	

function BeamSearch:run(x, maxLength)
    --[[
    Beam search:
    - x: source sentence
    - maxLength: maximum length of the translation
    ]]
    self.model:clearState()
    --self.model:resetStates()
    self.model.decoder._rnn.rememberStates = false 
    local K, Nw = self.K, self.Nw

    local x = encodeString(x, self.vocab[1], self.reverseInput)
    -- not that if we do this, the first prediction will be the same
    x = x:repeatTensor(K, 1)
    local srcLength = x:size(2)
    local T = maxLength or utils.round(srcLength * 1.4)
    self.model:stepEncoder(x)

    local hypos = torch.CudaLongTensor(K, T):fill(self.bosidx)
    local nbestCands = {}
    local nbestScores = {}

    local scores = torch.CudaTensor(K, 1):zero()
    local alpha = self.normLength
    for t = 1, T-1 do
        local curIdx = hypos[{{}, {t}}]
	print(curIdx)
        local logProb = self.model:stepDecoder(curIdx)
        -- quick hack to handle the first prediction
        if t == 1 then
            logProb[{{2, K}, {}}]:fill(-math.huge)
        end
        local maxscores, indices = logProb:topk(Nw, true) -- prune here
    ----
    --
    --
    --
    --
--    maxscores, indices = prune(maxscores, indices)
        --self.aggrEntr = self.aggrEntr  + entropy(maxscores)
	--self.numPred = self.numPred + maxscores:size(1)
        local curscores = scores:repeatTensor(1,Nw) 
        maxscores:add(curscores)

        local _scores, flatIdx = maxscores:view(-1):topk(K, true)
        scores = _scores:view(-1, 1)
        local nr, nc = maxscores:size(1), maxscores:size(2) -- K , Nw
        -- TODO: check with CudaLongTensor
        local rowIdx = flatIdx:long():add(-1):div(nc):add(1):typeAs(hypos)
        local colIdx = indices:view(-1):index(1, flatIdx)

        local xc = colIdx:eq(self.eosidx)--:type('torch.CudaLongTensor') -- completed sentence
        local nx = xc:sum()--:type('torch.CudaLongTensor')
        if nx > 0 then
            -- found some candidates
            --scores = scores:float()
            local cands = rowIdx:maskedSelect(xc)
            local completedHyps = hypos:index(1, cands):float():narrow(2, 1, t)
	         local xscores = scores:maskedSelect(xc):view(-1)
            --local xscores = scores:index(1, xc:type('torch.CudaLongTensor')):view(-1)
	    --print(xscores:size())
            -- add to nbest
            xscores:div(t^alpha)
            for i = 1, nx do
                local text = decodeString(completedHyps[i], self.vocab[2].idx2word, self._ignore)
                local s = xscores[i]
                table.insert(nbestCands, text)
                table.insert(nbestScores, s)
            end

            if nx == colIdx:numel() then break end
            -- check remaining in-completed hypotheses
            local rh = colIdx:ne(self.eosidx)
            rowIdx = rowIdx:maskedSelect(rh)
            colIdx = colIdx:maskedSelect(rh)
            scores = scores:maskedSelect(rh)
            -- reduce K
            K = rh:sum()
        end

        -- keep survival hypotheses
        hypos = hypos:index(1, rowIdx)
        hypos[{{}, t+1}] = colIdx
        self.model:indexStates(rowIdx)
    end
    if #nbestCands == 0 then
        assert(K == self.K)
        scores = scores:view(-1)
        scores:div(T^alpha)
        for i = 1, K do
            local text = decodeString(hypos[i], self.vocab[2].idx2word, self._ignore)
            table.insert(nbestCands, text)
            table.insert(nbestScores, scores[i])
        end
    end
    local _, idx = torch.Tensor(nbestScores):topk(1, true)
    return nbestCands[idx[1]]
end
