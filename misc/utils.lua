local utils = {}
local _ = require 'moses'

function utils.round(num, idp)
    local mult = 10^(idp or 0)
    return math.floor(num * mult + 0.5) / mult
end

function utils.encodeString(input, vocab, reverse)
    -- map a sentence to a tensor of idx
    local xs = stringx.split(input)
    if reverse then
        xs = _.reverse(xs)
    end
    local ids = {}
    for _, w  in ipairs(xs) do
        local idx = vocab[w] or vocab['<unk>']
        table.insert(ids, idx)
    end
    return torch.Tensor(ids):view(1, -1)
end

function utils.decodeString(x, id2word, _ignore)
    --[[ Map tensor if indices to surface words

    Parameters:
    - `x` : tensor input, 1D for now
    - `id2word` : mapping dictionary
    - `_ignore` : dictionary of ignored words such as <s>, </s>

    Return:
    - `s` : string
    -]]
    local words = {}
    for i = 1, x:numel() do
        local idx = x[i]
        if not _ignore[idx] then
            table.insert(words, id2word[idx])
        end
    end
    return table.concat(words, ' ')
end


function utils.flat_to_rc(v, indices, flat_index)
    -- from flat tensor recover row and column index of an element
    local row = math.floor((flat_index - 1)/v:size(2)) + 1
    return row, indices[row][(flat_index - 1) % v:size(2) + 1]
end

function utils.topk(k, mat, index)
    --[[ find top k elements in the matrix

    Parameters:
    - `k` : number of elements
    - `mat` : a matrix
    - `index: a matrix of LongTensor, same size as mat and store corresponding index
    --]]
    local res, flat_idx = mat:view(-1):topk(k, true)

    if mat:type() == 'torch.CudaTensor' then
        flat_idx = flat_idx:long() -- need long tensor here
    end

    flat_idx:add(-1)
    local dim2 = mat:size(2)
    local row = flat_idx:clone():div(dim2):add(1)
    local _idx = flat_idx:mod(dim2):add(1):view(-1,1):typeAs(mat)
    local col = index:index(1, row):gather(2, _idx)

    if mat:type() == 'torch.CudaTensor' then
        row = row:type('torch.CudaTensor')
    else
        col = col:long()
    end

    return res, row, col
end


function utils.find_topk(k, mat)
    --[[ find top k elements in the matrix

    Parameters:
    - `k` : number of elements
    - `mat` : a matrix

    Return:
    - `value` : k values
    - `row` : corresponding row
    - `col` : corresponding column
    --]]
    local res, idx = mat:view(-1):topk(k, true)
    local dim2 = mat:size(2)

    idx:add(-1)
    local row = idx:clone():div(dim2):add(1)
    local col = idx:mod(dim2):add(1)
    return res, row, col
end

function utils.extractCorrectPredictions(logProbTensor,targetTensor,labelValue,correctBeam)
	local highestVals,predictions = logProbTensor:topk(correctBeam,true)
	if labelValue == 'binary' then
		if correctBeam == 1 then 
			return torch.eq(predictions,targetTensor):cuda()
		else
			local inBeam = torch.zeros(targetTensor:size(1),1)
			for i=1,targetTensor:size(1) do
				local corr = targetTensor[i]
				for j=1,correctBeam do
					if predictions[i][j] == corr then inBeam[i][1] = 1 end
				end
			end
			return inBeam:cuda()
		end
	elseif labelValue == 'relativDiff' then
		local correct = torch.CudaTensor(highestVals:size())
		for i=1,targetTensor:size(1) do
			correct[i][1]=logProbTensor[i][targetTensor[i]]
		end
		local result = torch.exp(torch.csub(correct,highestVals))
		return result
	end	
end	

function utils.extractBeamRegionOfCorrect(logProbTensor,targetTensor,classes)
    local topDistr,ind = logProbTensor:topk(classes[#classes],true)
    local foundIndex = torch.Tensor(logProbTensor:size(1)):fill(#classes+1)
    local classes = classes
    
    function getClass(index)
        for i,cl in ipairs(classes) do
            if index == cl or index < cl then return i end
        end
    end

    for i=1,ind:size(1) do
        local t = targetTensor[i]
        for j=1,ind:size(2) do
            if ind[i][j] == t then
                foundIndex[i] = getClass(j)
                break
            end
        end
    end
    return foundIndex
end 

function turnIntoCumulativeTarget(classVector,numClasses)
    local result = torch.Tensor(classVector:size(1),numClasses):fill(0)
    for i=1,classVector:size(1) do
        for j=classVector[i],numClasses do
            result[i][j]=1
        end
    end
    return result:cuda()

end

function utils.reverse(t, dim)
    --[[ Reverse tensor along the specified dimension

    Parameters:
    - `t` : tensor to be reversed
    - `dim` : integer 1 or 2

    Return:
    - `rev_t` : reversed tensor
    --]]
    local dtype = 'torch.LongTensor'
    if t:type() == 'torch.CudaTensor' then dtype = 'torch.CudaTensor' end

    local rev_idx
    if not dim then
        -- this is a 1 D tensor
        rev_idx = torch.range(t:numel(), 1, -1):type(dtype)
        return t:index(1, rev_idx)
    end

    assert(t:dim() == 2)
    if t:size(d) == 1 then return t:clone() end

    rev_idx = torch.range(t:size(dim), 1, -1):type(dtype)

    return t:index(dim, rev_idx)
end

function utils.scale_clip(v, max_norm)
    local norm = v:norm()
    if norm > max_norm then
        v:div(norm/max_norm)
    end
end

return utils
