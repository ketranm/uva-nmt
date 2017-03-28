require 'cutorch'
require 'nn'
require 'cunn'
require 'tardis.EnsemblePrediction'
require 'data.loadMultiText'
local _ = require 'moses'
local cfg = require 'pl.config'
local timer = torch.Timer()
torch.manualSeed(42)
local cfg = require 'pl.config'
local opt = cfg.read(arg[1])
local multiOpt = {}
for c in opt.configFiles:gmatch("%S+") do
    local indivOpt =  cfg.read(c)
    indivOpt.padIdx = 1
    table.insert(multiOpt,indivOpt)
end
if not opt.gpuid then opt.gpuid = 0 end
torch.manualSeed(opt.seed or 42)
cutorch.setDevice(opt.gpuid + 1)
cutorch.manualSeed(opt.seed or 42)

print('Experiment Setting: ', opt)
io:flush()

--statistics exrtacting functions

function computeTopKDistr(distr,k)
    local topPr,ind = distr:topk(k,true)
    return {topPr,ind}
end

function computeThresholdDistr(distr,thr)
    local thrTensor = torch.Tensor(distr:size()):fill(thr)
    local filter = torch.gt(distr,thrTensor)
    local topPr = torch.cmul(thrTensor,filter:float())
    local indicesTensor = torch.zeros(distr:size())
    for i=1,distr:size(2) do 
        indicesTensor[{{},{i}}]:fill(i)
    end
    indicesTensor:cmul(filter:float())
    return {topPr,indicesTensor}
end


function classOverlap(indices_1,indices_2)
    local intersectIndices_1 = {}
    local intersectIndices_2 = {}

    local numInstances = indices_1:size(1)
    local numClasses = indices_1:size(2)
    for i=1,numInstances do 
        local perInstanceIndices_1 = {}
        local perInstanceIndices_2 = {}
        for clInd_1=1,numClasses do
            local class_1 = indices_1[i][clInd_1]
            if class_1 ~= 0 then
                for clInd_2=1,numClasses do
                    if class_1 == indices_1[i][clInd_2] then 
                        table.insert(perInstanceIndices_1,clInd_1)
                        table.insert(perInstanceIndices_2,clInd_2)
                        break
                    end
                end
            end
        end
        table.insert(intersectIndices_1,perInstanceIndices_1)
        table.insert(intersectIndices_2,perInstanceIndices_2)
    end
    return {intersectIndices_1,intersectIndices_2}
end

function updateOverlap(k,intersectClass_12,intersectClass_1ens,intersectClass_2ens)
    local numIntersect = 0
    for i=1,#intersectClass_12 do
        for j=1,#intersectClass_12[i] do
            if intersectClass_12[i][j] > 0 then numIntersect = numIntersect + 1 end
        end
    end
    overlap_12[k] = overlap_12[k] + numIntersect

    numIntersect = 0
    for i=1,#intersectClass_1ens do
        for j=1,#intersectClass_1ens[i] do
            if intersectClass_1ens[i][j] > 0 then numIntersect = numIntersect + 1 end
        end
    end
    overlap_ens1[k] = overlap_ens1[k] + numIntersect

    numIntersect = 0
    for i=1,#intersectClass_2ens do
        for j=1,#intersectClass_2ens[i] do
            if intersectClass_2ens[i][j] > 0 then numIntersect = numIntersect + 1 end
        end
    end
    overlap_ens2[k] = overlap_ens2[k] + numIntersect
end

function updateCrossEntr(k,intersect,topDistr)
    local result_12 = 0
    local result_21 = 0

    function aggrCrossEntr(intersect_1,intersect_2,distr1,distr2)
        local result_12 = 0
        local result_21 = 0
        local prob1 = torch.exp(distr1)
        local prob2 = torch.exp(distr2)
        for i=1,#intersect_1 do
            for j=1,#intersect_1[i] do
                local ind1 = intersect_1[i][j]
                local ind2 = intersect_2[i][j]
                result_12 = result_12 + distr1[i][ind1]*prob2[i][ind2]
                result_21 = result_21 + distr2[i][ind2]*prob1[i][ind1]
            end
        end
        result_12 = result_12
        result_21 = result_21
        return {result_12,result_21}
    end

    -- 1,2
    local updates_12 = aggrCrossEntr(intersect[1][1],intersect[1][2],topDistr[1],topDistr[2])
    crossEntr_12[k][1] = crossEntr_12[k][1] - updates_12[1]
    crossEntr_12[k][2] = crossEntr_12[k][2] - updates_12[2]

    -- 1,ens
    local updates_1ens = aggrCrossEntr(intersect[2][1],intersect[2][2],topDistr[1],topDistr[3])
    crossEntr_1ens[k][1] = crossEntr_1ens[k][1] - updates_1ens[1]
    crossEntr_1ens[k][2] = crossEntr_1ens[k][2] - updates_1ens[2]

    --2,ens
    local updates_2ens = aggrCrossEntr(intersect[3][1],intersect[3][2],topDistr[2],topDistr[3])
    crossEntr_2ens[k][1] = crossEntr_2ens[k][1] - updates_2ens[1]
    crossEntr_2ens[k][2] = crossEntr_2ens[k][2] - updates_2ens[2]

end

function updateEntr(k,topDistr)
    function aggrEntr(distr)
        local prob = torch.exp(distr)
        local result = torch.sum(torch.cmul(distr,prob)) * (-1)
        return result
    end
    entropy_1[k] = entropy_1[k] + aggrEntr(topDistr[1])
    entropy_2[k] = entropy_2[k] + aggrEntr(topDistr[2])
    entropy_ens[k] = entropy_ens[k] + aggrEntr(topDistr[3])
end

function updateCorrectInTop(k,indices,target)
    local target = target:view(-1)
    function findCorrect(ind)
        local timesCorrect = 0
        for i=1,ind:size(1) do
            local targetClass = target[i]
            for j=1,ind:size(2) do
                if ind[i][j] == targetClass then
                    timesCorrect = timesCorrect + 1
                    break
                end
            end
        end
        return timesCorrect
    end
    correctInTopK_1[k] = correctInTopK_1[k] + findCorrect(indices[1])
    correctInTopK_2[k] = correctInTopK_2[k] + findCorrect(indices[2])
    correctInTopK_ens[k] = correctInTopK_ens[k] + findCorrect(indices[3])
end

function updateAnalysisStructuresK(distributions,k,target,topFunction)
    local d1,d2,ens = unpack(distributions)
    local topDistr = _.map(distributions,function(i,v) return topFunction(v,k) end)

    local intersectClass_12 = classOverlap(topDistr[1][2],topDistr[2][2]) --DONE
    local intersectClass_1ens = classOverlap(topDistr[1][2],topDistr[3][2])
    local intersectClass_2ens = classOverlap(topDistr[2][2],topDistr[3][2])
    updateOverlap(k,intersectClass_12[1],intersectClass_1ens[1],intersectClass_2ens[1]) -- DONE
    updateCrossEntr(k,{intersectClass_12,intersectClass_1ens,intersectClass_2ens},_.map(topDistr,function(i,v) return v[1] end)) -- DONE
    updateEntr(k,_.map(topDistr,function(i,v) return v[1] end)) -- DONE
    updateCorrectInTop(k,_.map(topDistr,function(i,v) return v[2] end),target) --DONE
    return _.map(topDistr,function(i,v) return v[1] end)
end

function topKdistr(distr,k)
	local vals,ind = distr:topk(k,true)
	return {vals,ind}
end

function updateCorrectRank(distributions,target)
    local target = target:view(-1)
    function getCorrectRank(distr,target)
        local rankAggr = 0
        local _,ind = distr:topk(30000,true)
        for i=1,ind:size(1) do
            local targetClass = target[i]
            for j=1,ind:size(2) do
                if ind[i][j] ==targetClass then
                    rankAggr = rankAggr + j
                    break
                end
            end
        end
        return rankAggr
    end


end
function updateKindependentStructures(distributions,target)
    updateCorrectRank(distributions,target)
end

function prepro(input)
    local xs, y = unpack(input)
    local seqlen = y:size(2)
    -- make contiguous and transfer to gpu
    xs = _.map(xs,function(i,x) return x:contiguous():cudaLong() end)
    prev_y = y:narrow(2, 1, seqlen-1):contiguous():cudaLong()
    next_y = y:narrow(2, 2, seqlen-1):contiguous():cudaLong()  --batchsize x seqlen
    return xs, prev_y, next_y
end



-- class overlap betweek topk distributions
 overlap_12 = {}
overlap_ens1 = {}
overlap_ens2 = {}
--(cormalized by k) crossentropy between distributions (tuple {ab,ba})
crossEntr_12 = {}
crossEntr_1ens = {}
crossEntr_2ens = {}
-- (normalized by k) entropy of topk distribution 
entropy_1 = {}
entropy_2 = {}
entropy_ens = {}
-- whether correct class is in topk distribution
correctInTopK_1 = {}
correctInTopK_2 = {}
correctInTopK_ens = {}
--rank of correct class (independent of topK) (dataset -- will be used for correlation)
correctRank_1 = {}
correctRank_2 = {}
correctRank_ens = {}
--loss (p_max - p_corr)/p_max  (independent of topK)
maxVScorrLoss_1 = {}
maxVScorrLoss_2 = {}
maxVScorrLoss_ens = {}



--parameterize top-distribution
local K = {5,20,100}--,1000}
local probThresholds = _.map({0.1,0.05,0.01},function(i,v) return torch.log(v) end)


--initialize analysis data structures
for _,k in ipairs(K) do
    overlap_12[k] = 0
    overlap_ens1[k] = 0
    overlap_ens2[k] = 0
    crossEntr_12[k] = {0,0}
    crossEntr_1ens[k] = {0,0}
    crossEntr_2ens[k] = {0,0}
    entropy_1[k] = 0
    entropy_2[k] = 0
    entropy_ens[k] = 0
    correctInTopK_1[k] = 0
    correctInTopK_2[k] = 0
    correctInTopK_ens[k] = 0
end

--[[
for _,thr in ipairs(probThresholds) do
    overlap_12[thr] = 0
    overlap_ens1[thr] = 0
    overlap_ens2[thr] = 0
    crossEntr_12[thr] = {0,0} --{(p1,p2),(p2,p1)}
    crossEntr_1ens[thr] = {0,0}
    crossEntr_2ens[thr] = {0,0}
    entropy_1[thr] = 0
    entropy_2[thr] = 0
    entropy_ens[thr] = 0
    correctInTopK_12[thr] = {0,0}
    correctInTopK_1ens[thr] = {0,0}
    correctInTopK_2ens[thr] = {0,0}
end
]]--




local ensemble = EnsemblePrediction(opt,multiOpt)
local loader = MultiDataLoader(opt,multiOpt)

local totalObservations = 0
loader:train()

local nbatches = loader.nbatches
print('nbatches:: '..nbatches)
for i=1,nbatches do
    print('batch '..i)
    local xs, prev_y, next_y = prepro(loader:next_origOrder())
    local distr_1,distr_2,distr_ens = ensemble:forwardAndOutputDistributions(xs,prev_y) --ComputeOutputOverlap(xs,prev_y,K)
    --updateKindependentStructures({distr_1,distr_2,distr_ens},next_y)

    local curr_distribs = {distr_1,distr_2,distr_ens}
    for i=#K,1,-1 do
        local newDistr = updateAnalysisStructuresK(curr_distribs,K[i],next_y,topKdistr) -- TODO: add topK function (for thresholds), for now just topK
        curr_distribs = newDistr
    end
    totalObservations = totalObservations + next_y:size(1)*next_y:size(2)
    --[[curr_distribs = {distr_1,distr_2,distr_ens}
    for i=#probThresholds,1,-1 do
        updateAnalysisStructuresThr(curr_distribs,probThresholds[i])
    end]]--
    
end



for _,k in ipairs(K) do
    local denom = totalObservations * k
    print('=======================')
    print('topK:: '..k)
    local classOverlap_12_k = overlap_12[k]/denom
    local classOverlap_ens1_k = overlap_ens1[k]/denom
    local classOverlap_ens2_k = overlap_ens2[k]/denom
    local crossEntr_12_k = {crossEntr_12[k][1]/denom,crossEntr_12[k][2]/denom}
    local crossEntr_1ens_k = {crossEntr_1ens[k][1]/denom,crossEntr_1ens[k][2]/denom}
    local crossEntr_2ens_k = {crossEntr_2ens[k][1]/denom,crossEntr_2ens[k][2]/denom}
    local entropy_1_k = entropy_1[k]/totalObservations
    local entropy_2_k = entropy_2[k]/totalObservations
    local entropy_ens_k = entropy_ens[k]/totalObservations
    local correctInTop_1_k = correctInTopK_1[k]/totalObservations
    local correctInTop_2_k = correctInTopK_2[k]/totalObservations
    local correctInTop_ens_k = correctInTopK_ens[k]/totalObservations
    print('class overlap 1 2::'..classOverlap_12_k)
    print('class overlap 1 ens::'..classOverlap_ens1_k)
    print('class overlap 2 ens::'..classOverlap_ens2_k)
    
    print('normalized crossEntr 1 2::'..crossEntr_12_k[1]..' '..crossEntr_12_k[2])
    print('normalized crossEntr 1 ens::'..crossEntr_1ens_k[1]..' '..crossEntr_1ens_k[2])
    print('normalized crossEntr 2 ens::'..crossEntr_2ens_k[1]..' '..crossEntr_2ens_k[2])

    print('entropy 1::'..entropy_1_k)
    print('entropy 2::'..entropy_2_k)
    print('entropy ens::'..entropy_ens_k)

    print('normalized number of correct in top 1::'..correctInTop_1_k)
    print('normalized number of correct in top 2::'..correctInTop_2_k)
    print('normalized number of correct in top ens::'..correctInTop_ens_k)
end

