require 'cutorch'
require 'nn'
require 'cunn'
require 'tardis.EnsemblePrediction'
require 'data.loadMultiText'
require 'tardis.topKDistribution'
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

    local numIntersect = 0
    for i=1,#intersectClass_1ens do
        for j=1,#intersectClass_1ens[i] do
            if intersectClass_1ens[i][j] > 0 then numIntersect = numIntersect + 1 end
        end
    end
    overlap_ens1[k] = overlap_ens1[k] + numIntersect

    local numIntersect = 0
    for i=1,#intersectClass_2ens do
        for j=1,#intersectClass_2ens[i] do
            if intersectClass_2ens[i][j] > 0 then numIntersect = numIntersect + 1 end
        end
    end
    overlap_ens2[k] = overlap_ens2[k] + numIntersect
end

function updateCrossEntr(k,topIndices,distr)
    local result_12 = 0
    local result_21 = 0

    function aggrCrossEntr(ind_1,ind_2,distr1,distr2)
        local result_12 = {0,0}
        local result_21 = {0,0}
        for i=1,ind_1:size(1) do
            local totalMass1 = torch.exp(totalLogMass(distr1[i]))
            local totalMass2 = torch.exp(totalLogMass(distr2[i]))
            local prob1 = torch.exp(distr1[i])
            local prob2 = torch.exp(distr2[i])
            local result_12_i = 0
            local result_21_i = 0
            for j=1,ind_1:size(2) do
                local i1 = ind_1[i][j]
                local i2 = ind_2[i][j]
                result_12_i = result_12_i + prob1[i1]*distr2[i1]
                result_21_i = result_21_i + prob2[i2]*distr1[i2]
            end
            result_12[1] = result_12[1] + result_12_i
            result_21[1] = result_21[1] + result_21_i
            
            result_12[2] = result_12[2] + result_12_i/totalMass1
            result_21[2] = result_21[2] + result_21_i/totalMass2
        end
        return {result_12,result_21}
    end


    -- 1,2
    local updates_12 = aggrCrossEntr(topIndices[1],topIndices[2],distr[1],distr[2])
    local updates_1ens = aggrCrossEntr(topIndices[1],topIndices[3],distr[1],distr[3])
    local updates_2ens = aggrCrossEntr(topIndices[2],topIndices[3],distr[2],topDistr[3])
    for dir=1,2 do
        for norm=1,2 do
            crossEntr_12[k][dir][norm] = crossEntr_12[k][dir][norm] - updates_12[dir][norm]
            crossEntr_1ens[k][dir][norm] = crossEntr_1ens[k][dir][norm] - updates_1ens[dir][norm]
            crossEntr_2ens[k][dir][norm] = crossEntr_2ens[k][dir][norm] - updates_2ens[k][dir][norm]
        end
    end

end

function updateEntr(k,topDistr)
    function aggrEntr(distr)
        local totalLogMass = totalLogMass(distr)
        local normProb = torch.exp(distr):div(torch.exp(totalLogMass))
        local normLogProb = torch.sub(distr,totalLogMass)
        local resultUnnorm = torch.sum(torch.cmul(normLogProb,normProb)) * (-1)
        local resultNorm = torch.sum(torch.cmul(distr,torch.exp(distr))) * (-1)
        return resultUnnorm,resultNorm
    end
    local entr1_unn, entr1_norm = aggrEntr(topDistr[1])
    entropy_1[k][1] = entropy_1[k][1] + entr1_unn
    entropy_1[k][2] = entropy_1[k][2] + entr1_norm

    local entr2_unn, entr2_norm = aggrEntr(topDistr[2])
    entropy_2[k][1] = entropy_2[k][1] + entr2_unn
    entropy_2[k][2] = entropy_2[k][2] + entr2_norm
    
    local entrens_unn, entrens_norm = aggrEntr(topDistr[3])
    entropy_ens[k][1] = entropy_ens[k][1] + entrens_unn
    entropy_ens[k][2] = entropy_ens[k][2] + entrens_norm
    
end

function updateTotalMassPerBeam(k,topDistr)
    totalMassPerBeam_1[k] = totalMassPerBeam_1[k] + totalLogMass(topDistr[1])
    totalMassPerBeam_2[k] = totalMassPerBeam_2[k] + totalLogMass(topDistr[2])
    totalMassPerBeam_ens[k] = totalMassPerBeam_ens[k] + totalLogMass(topDistr[3])
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
    updateCrossEntr(k,_.map(topDistr,function(i,v) return v[2] end),distributions)
    updateEntr(k,_.map(topDistr,function(i,v) return v[1] end)) -- DONE
    updateCorrectInTop(k,_.map(topDistr,function(i,v) return v[2] end),target) --DONE
    updateTotalMassPerBeam(k,_.map(topDistr,function(i,v) return v[1] end))
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



totalMassPerBeam_1 = {}
totalMassPerBeam_2 = {}
totalMassPerBeam_ens = {}
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
    entropy_1[k] = {0,0}
    entropy_2[k] = {0,0}
    entropy_ens[k] = {0,0}
    correctInTopK_1[k] = 0
    correctInTopK_2[k] = 0
    correctInTopK_ens[k] = 0
    totalMassPerBeam_1[k] = 0
    totalMassPerBeam_2[k] = 0
    totalMassPerBeam_ens[k] = 0
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
    local classOverlap_12_k = overlap_12[k]/totalObservations
    local classOverlap_ens1_k = overlap_ens1[k]/totalObservations
    local classOverlap_ens2_k = overlap_ens2[k]/totalObservations
    local crossEntr_12_k_unn = {crossEntr_12[k][1][1]/denom,crossEntr_12[k][2][1]/denom}
    local crossEntr_12_k_norm = {crossEntr_12[k][1][2]/totalObservations,crossEntr_12[k][2][2]/totalObservations}
    local crossEntr_1ens_k_unn = {crossEntr_1ens[k][1][1]/denom,crossEntr_1ens[k][2][1]/denom}
    local crossEntr_1ens_k_norm = {crossEntr_1ens[k][1][2]/totalObservations,crossEntr_1ens[k][2][2]/totalObservations}
    local crossEntr_2ens_k_unn = {crossEntr_2ens[k][1][1]/denom,crossEntr_2ens[k][2][1]/denom}
    local crossEntr_2ens_k_unn = {crossEntr_2ens[k][1][2]/totalObservations,crossEntr_2ens[k][2][2]/totalObservations}
    local entropy_1_k_unn = entropy_1[k][1]/denom
    local entropy_1_k_norm = entropy_1[k][2]/totalObservations
    local entropy_2_k_unn = entropy_2[k][1]/denom
    local entropy_2_k_norm = entropy_2[k][2]/totalObservations
    local entropy_ens_k_unn = entropy_ens[k][1]/denom
    local entropy_ens_k_norm = entropy_ens[k][2]/totalObservations
    local correctInTop_1_k = correctInTopK_1[k]/totalObservations
    local correctInTop_2_k = correctInTopK_2[k]/totalObservations
    local correctInTop_ens_k = correctInTopK_ens[k]/totalObservations

    local logMass_k_1 = totalMassPerBeam_1[k]/totalObservations
    local logMass_k_2 = totalMassPerBeam_2[k]/totalObservations
    local logMass_k_ens = totalMassPerBeam_ens[k]/totalObservations

    print('total log.mass per beam 1::'..logMass_k_1)
    print('total log.mass per beam 2::'..logMass_k_2bv)
    print('total log.mass per beam ens::'..logMass_k_ens)

    print('unnorm.crossEntr 1 2::'..crossEntr_12_k_unn[1]..' '..crossEntr_12_k_unn[2])
    print('unnorm.crossEntr 1 ens::'..crossEntr_1ens_k_unn[1]..' '..crossEntr_1ens_k_unn[2])
    print('unnorm.crossEntr 2 ens::'..crossEntr_2ens_k_unn[1]..' '..crossEntr_2ens_k_unn[2])

    print('normalized crossEntr 1 2::'..crossEntr_12_k_norm[1]..' '..crossEntr_12_k_norm[2])
    print('normalized crossEntr 1 ens::'..crossEntr_1ens_k_norm[1]..' '..crossEntr_1ens_k_nomr[2])
    print('normalized crossEntr 2 ens::'..crossEntr_2ens_k_norm[1]..' '..crossEntr_2ens_k_norm[2])

    print('unnorm.entropy 1::'..entropy_1_k_unn)
    print('unnorm.entropy 2::'..entropy_2_k_unn)
    print('unnorm.entropy ens::'..entropy_ens_k_unn)

    print('norm.entropy 1::'..entropy_1_k_norm)
    print('norm.entropy 2::'..entropy_2_k_norm)
    print('norm.entropy ens::'..entropy_ens_k_norm)

    print('class overlap 1 2::'..classOverlap_12_k)
    print('class overlap 1 ens::'..classOverlap_ens1_k)
    print('class overlap 2 ens::'..classOverlap_ens2_k)

    print('normalized number of correct in top 1::'..correctInTop_1_k)
    print('normalized number of correct in top 2::'..correctInTop_2_k)
    print('normalized number of correct in top ens::'..correctInTop_ens_k)
end

