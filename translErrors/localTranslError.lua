function inSet(s,elem) 
	for _,e1 in ipairs(s) do
		if elem == e1 then return true end
	end
	return false
end

function yule_q(errorVector,agreeProd,disagreeProd)
	return (agreeProd - disagreeProd)/(agreeProd + disagreeProd)
end

function rho_correlation(errorVector,agreeProd,disagreeProd)
		local errorFirst = errorVector[1] + errorVector[3]
	local errorSecond = errorVector[1] + errorVector[4]
	local correctFirst = errorVector[2] + errorVector[4]
	local correctSecond = errorVector[2] + errorVector[3]

	return (agreeProd + disagreeProd)/(errorFirst * errorSecond * correctFirst * correctSecond)
end

function disagreementMeasure(errorVector,agreeProd,disagreeProd)
		return (errorVector[3] + errorVector[4])/(agreeProd + disagreeProd)
end

function doubleFault(errorVector,agreeProd,disagreeProd)
		return errorVector[2]/(agreeProd + disagreeProd)
end

function computePairwiseStatistics(errorVector)
	local agreeProd = torch.float(errorVector[1]*errorVector[2]
	local disagreeProd = errorVector[3]*errorVector[4]
	local result = {}
	result['yule_q'] =  yule_q(errorVector,agreeProd,disagreeProd)
	result['rho_correlation'] = rho_correlation(errorVector,agreeProd,disagreeProd)
	result['disagreementMeasure'] = disagreementMeasure(errorVector,agreeProd,disagreeProd)
	result['doubleFault'] = doubleFault(errorVector,agreeProd,disagreeProd)
	return result
end 

function initializePairwiseStatistics()
	local result = {}
	result['yule_q'] =  0
	result['rho_correlation'] = 0
	result['disagreementMeasure'] = 0
	result['doubleFault'] = 0
	return result
end



local LocalTranslError = torch.class('LocalTranslError')

function LocalTranslError:__init(numberErrorInputTypes) 
	self.errorObservations = torch.zeros(numberErrorInputTypes)
end

function LocalTranslError:extractError(errorInput)
	self.errorObservations:add(errorInput)
	--table.insert(self.errorObservations,{sentN,positN})
end

local BinaryTranslError,parent = torch.class('torch.BinaryTranslError','LocalTranslError')

function BinaryTranslError:__init(numberErrorInputTypes)
        self.errorObservations = torch.zeros(numberErrorInputTypes)
end

function BinaryTranslError:reset()
	self.errorObservations=torch.zeros(self.errorObservations:size(1))
end

function BinaryTranslError:extractError(sentN,positN)
        self.errorObservations:add(errorInput)
end

function BinaryTranslError:computePairwiseStatistics()
	self.pairwiseStaitistics = computePairwiseStaitistics(self.errorObservations)
	return self.pairwiseStaitistics
end


local BinaryTranslError,parent = torch.class('torch.BinaryTranslError','LocalTranslError')

function BinaryTranslError:__init() 
	self.errorObservations = {}
end

function BinaryTranslError:extractError(sentN,positN)
	table.insert(self.errorObservations,{sentN,positN})
end

local RefWordTranslError, parent = torch.class('torch.RefWordTranslError', 'LocalTranslError')

function RefWordTranslError:__init(vocabulary,metaSymbols,numberErrorInputTypes)
	self.errorObservations = {}
	for _,v in ipairs(vocabulary) do
		if not inSet(metaSymbols,v) then
			self.errorObservations[v] = torch.zeros(numberErrorInputTypes)
		end
	end
end

function RefWordTranslError:reset()
	for v,counts in pairs(self.errorObservations) do
			self.errorObservations[v] = torch.zeros(counts:size(1))
	end
end


function RefWordTranslError:extractError(errorInput,refSequenceVector)
	local errorClass = refSequenceVector[positN]
	self.errorObservations[errorClass]:add(errorInput)
end

function RefWordTranslError:computePairwiseStatistics()
	totalNumberObservations = 0
	result = initializePairwiseStatistics()

	for cl,errVector in pairs(self.errorObservations) do
		numberClassObservations = errVector:sum()
		totalNumberObservations = totalNumberObservations + numberClassObservations
		classPairwiseStatistics = computePairwiseStaitistics(errVector)
		for stat,value in pairs(classPairwiseStatistics) do
			result[stat] = result[stat] + numberClassObservations*value
		end
	end
	for stat,value in pairs(result) do
		result[stat] = result[stat]/totalNumberObservations
	end
	self.pairwiseStaitistics = result
	return self.pairwiseStaitistics
end



-- v1 is correct, v2 is prediction
local ConfusionTranslError, parent = torch.class('torch.ConfusionTranslError', 'LocalTranslError')

function ConfusionTranslError:__init(vocabulary,metaSymbols,numberErrorInputTypes)
	self.errorObservations = {}
		for _,v1 in ipairs(vocabulary) do
			if not inSet(metaSymbols,v1) then
				self.errorObservations[v1] = {}
				for _,v2 in ipairs(vocabulary) do
					if not inSet(metaSymbols,v2) then
						self.errorObservations[v1][v2] = torch.zeros(numberErrorInputTypes)
					end
				end
			end
		end
end

function ConfusionTranslError:extractError(errorInput,positN,refSequenceVector,predSequenceVector)
	local correctVal = refSequenceVector[positN]
	local predVal = predSequenceVector[positN]
	self.errorObservations[correctVal][predVal]:add(errorInput)
end


local NgramContextTranslError, parent = torch.class('torch.NgramContextTranslError', 'LocalTranslError')

function NgramContextTranslError:__init(vocabulary,metaSymbols,ngramOrder,numberErrorInputTypes)
	self.contextLen = ngramOrder 
	self.errorObservations = {}
	self.numberErrorInputTypes = numberErrorInputTypes
	for _,v in ipairs(vocabulary) do
		if not inSet(metaSymbols,v1) then
			self.errorObservations[v] = {}
		end
	end
end

function NgramContextTranslError:reset()
	for v,_ in pairs(self.errorObservations) do
		self.errorObservations[v] = {}
	end
end

function NgramContextTranslError:extractError(errorInput,positN, refSequenceVector)
	--local errorVal = refSequenceVector[positN]
	local currTable = self.errorObservations--[errorVal]
	local currPositN = positN

	for i=1,self.contextLen do
		if currPositN < 1 then break end
		local currValue = refSequenceVector[currPositN]
		if currTable[currValue] == nil then currTable[currValue] = {} end
		currTable = currTable[currValue]
		currPositN = currPositN - 1
	end
	if currTable[-1] == nil then currTable[-1] = torch.zeros(self.numberErrorInputTypes) end
	currTable[-1]:add(errorInput)
end
	
	
function RefWordTranslError:computePairwiseStatistics()
	totalNumberObservations = 0

	function traverseTree(tree)
		totalNumberObservations = 0
		result = initializePairwiseStatistics()		
		for cl,subtree in pairs(tree) do
			if cl == -1 then
				numberClassObservations = subtree:sum()
				classStatistics = computePairwiseStaitistics(subtree)
				for stat,value in pairs(classStatistics)
					result[stat] = result[stat] + value*numberClassObservations
				end
				totalNumberObservations = totalNumberObservations + numberClassObservations
			else 
				local subreeNumberObservations, subtreeStatistics = traverseTree(subtree)
				totalNumberObservations = totalNumberObservations + subreeNumberObservations
				for stat,value in pairs(subtreeStatistics) do
					result[stat] = result[stat] + value
				end
			end
		end
		return totalNumberObservations, result
	end

	local totalNumberObservations,unNormalizedStatistics = traverseTree(self.errorObservations)
	self.pairwiseStaitistics = {}
	for stat,value in pairs(unNormalizedStatistics ) do
		self.pairwiseStaitistics[stat] = unNormalizedStatistics[stat]/totalNumberObservations
	end
	
	return self.pairwiseStaitistics
end

	


