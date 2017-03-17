local EnsemblePrediction = torch.class('EnsemblePrediction')
local cfg = require 'pl.config'
require 'tardis.SeqAtt'
local _ = require 'moses'
require 'tardis.BeamSearch'
require 'tardis.ensembleCombination'


function EnsemblePrediction:__init(kwargs,multiKwargs)
    --combination parameters
    self.combinInput = kwargs.combinInput
    self.contextCombinInput = kwargs.contextCombinInput
    self.combinMethod = kwargs.combinMethod
    self.beamSize = kwargs.beamSize
    self.maxLength = kwargs.maxLength
    scalarWeights = {}
    for w in kwargs.scWeights:gmatch("%S+") do table.insert(scalarWeights,w) end

    --models parameters
    ------self.embeddingSize = kwargs.embeddingSize   
    self.trgVocabSize = kwargs.trgVocabSize
    self.models = {}
    self.configs = {}
    self.vocabs = {}
    self.numModels = 0
    for i,model_kwargs in ipairs(multiKwargs) do 
       self.numModels = self.numModels + 1
        local vocab = torch.load(model_kwargs.dataPath..'/vocab.t7')
        local m = nil
        if self.combinMethod == 'confidPrediction' then
            m = nn.confidentNMT(model_kwargs)
            m:type('torch.CudaTensor')
            m:loadModelWithoutConfidence(model_kwargs.modelToLoad)    
            m:loadConfidence(model_kwargs.confidenceModel)
            m.confdence:evaluate()
        else
            m = nn.NMT(model_kwargs)
            m:type('torch.CudaTensor')
            m:load(model_kwargs.modelFile)
        end
        --m:use_vocab(vocab)
        m:evaluate()
    
        table.insert(self.vocabs,vocab)
        table.insert(self.models,m)
        table.insert(self.configs,model_kwargs)
    end 
        
    --decoding parameters
    self.K = kwargs.beamSize or 10
    self.Nw = kwargs.Nw or self.K * self.K
    self.reverseInput = kwargs.reverseInput or true
--    self.normLength = kwargs.normLength or 0 -- 1 if we normalized scores by length

    self.bosidx = self.vocabs[1][2].word2idx['<s>']
    self.eosidx = self.vocabs[1][2].word2idx['</s>']
    self.unkidx = self.vocabs[1][2].word2idx['<unk>']
    self.padidx = self.vocabs[1][2].word2idx['<pad>']

    self._ignore = {[self.bosidx] = true, [self.eosidx] = true}
    self.normLength = kwargs.normLength or 1
    

    --combination function
    if self.combinMethod == 'scalar' then
        self.combinMachine = scalarCombination(scalarWeights,self.trgVocabSize,self.combinInput)
    elseif self.combinMethod == 'entropyConfidence' then
        self.combinMachine = entropyConfidence()
    elseif self.combinMethod == 'entropyConfidenceBinary' then
        self.combinMachine = entropyConfidenceBinary()
    elseif self.combinMethod == 'entropyConfidenceBinaryReverse' then
        self.combinMachine = entropyConfidenceBinaryReverse()
    elseif self.combinMethod == 'scalarRandom' then
        self.combinMachine = combinMachine.randomScalarCombination(self.combinInput)
    elseif self.combinMethod == 'loglinCombination' then
        self.combinMachine = combinMachine.loglinCombination(kwargs.combMachineParametersFile,self.numModels)
        --TODO: do other options as well
    elseif self.combinMethod == 'expertMixture' then
        self.combinMachine = combinMachine.expertMixture(kwargs.combMachineParametersFile,self.embeddingSize,kwargs.combinWithBackprop)
    elseif self.combinMethod == 'expertMixtureHier' then
        self.combinMachine = combinMachine.expertMixture(kwargs.combMachineParametersFile,kwargs.combMachineSubmodel1,kwargs.combMachineSubmodel2,
                    self.embeddingSize,kwargs.combinWithBackprop)
    elseif self.combinMethod == 'confidPrediction' then
        self.combinMachine = combinMachine.confidenceMixture(kwargs.confidenceScoreCombination)
    end

    return self
end


function EnsemblePrediction:translate(xs)
    for i,m in ipairs(self.models) do 
        m:clearState() 
        m.decoder._rnn.rememberStates = false 
    end
    local K, Nw = self.K, self.Nw
    local xs = _.map(xs, function(i,x) return encodeString(x, self.vocabs[i][1], self.reverseInput) end)
    -- not that if we do this, the first prediction will be the same
    xs = _.map(xs,function(i,x) return x:repeatTensor(K, 1) end)
    local srcLengths = _.map(xs,function(i,x) return x:size(2) end)
    local T =  self.maxLength --  or utils.round(srcLength * 1.4) TODO functon of all source sentences?
    for i,x in ipairs(xs) do
	self.models[i]:stepEncoder(x) end

    local hypos = torch.CudaLongTensor(K, T):fill(self.bosidx)
    local nbestCands = {}
    local nbestScores = {}
    local scores = torch.CudaTensor(K, 1):zero()
    local alpha = self.normLength
    
    for t = 1, T-1 do
        local curIdx = hypos[{{}, {t}}] -- t-th slice (column size K)
        local logProb = self:decodeAndCombinePredictions(curIdx,t)
	--print(logProb[1])
        local maxscores, indices = logProb:topk(Nw, true)
        local curscores = scores:repeatTensor(1, Nw)
        maxscores:add(curscores)

        local _scores, flatIdx = maxscores:view(-1):topk(K, true)
        scores = _scores:view(-1, 1)
        local nr, nc = maxscores:size(1), maxscores:size(2)
        local rowIdx = flatIdx:long():add(-1):div(nc):add(1):typeAs(hypos)
        local colIdx = indices:view(-1):index(1, flatIdx)

        local xc = colIdx:eq(self.eosidx)--:type('torch.CudaLongTensor') -- completed sentence
        local nx = xc:sum()--:type('torch.CudaLongTensor')
        if nx > 0 then
             -- found some candidates
            local cands = rowIdx:maskedSelect(xc)
            local completedHyps = hypos:index(1, cands):float():narrow(2, 1, t)
            local xscores = scores:maskedSelect(xc):view(-1)

            -- add to nbest
            xscores:div(t^alpha)
            for i = 1, nx do
                local text = decodeString(completedHyps[i], self.vocabs[1][2].idx2word, self._ignore)
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
        for i,m in ipairs(self.models) do m:indexStates(rowIdx) end
    end

    if #nbestCands == 0 then
        assert(K == self.K)
        scores = scores:view(-1)
        scores:div(T^alpha)
        for i = 1, K do
            local text = decodeString(hypos[i], self.vocabs[1][2].idx2word, self._ignore)
            table.insert(nbestCands, text)
            table.insert(nbestScores, scores[i])
        end
    end
    local _, idx = torch.Tensor(nbestScores):topk(1, true)
    return nbestCands[idx[1]]

end


function EnsemblePrediction:decodeAndCombinePredictions(curIdx,timeStep)
    local logProbs = nil
    local combinWeights = nil
    if self.combinMethod == 'confidPrediction' then
        _.each(self.models, function(i,m) m:stepDecoderUpToHidden(curIdx) end)
        logProbs = _.map(self.models, function(i,m) return m:predictTargetLabel() end)
        combinWeights = _.map(self.models,function(i,m) return m:predictConfidenceScore() end)
    else
        logProbs = _.map(self.models, function(i,m) return m:stepDecoder(curIdx) end) 
    end
    -- quick hack to handle the first prediction
    for i,lPr in ipairs(logProbs) do
        if t == 1 then
            lPr[{{2, K}, {}}]:fill(-math.huge) -- gives <pad> highest prob - why??
        end
    end
    return self.combinMachine(logProbs,combinWeights)

end
