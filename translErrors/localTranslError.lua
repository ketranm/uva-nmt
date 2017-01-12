function inSet(s,elem) 
	for _,e1 in ipairs(s) do
		if elem == e1 then return true end
	end
	return false
end



local LocalTranslError = torch.class('LocalTranslError')

function LocalTranslError:__init() 
	self.errorObservations = {}
end

function LocalTranslError:extractError(sentN,positN)
	table.insert(self.errorObservations,{sentN,positN})
end


local BinaryTranslError,parent = torch.class('torch.BinaryTranslError','LocalTranslError')

function BinaryTranslError:__init() 
	self.errorObservations = {}
end

function BinaryTranslError:extractError(sentN,positN)
	table.insert(self.errorObservations,{sentN,positN})
end

local RefWordTranslError, parent = torch.class('torch.RefWordTranslError', 'LocalTranslError')

function RefWordTranslError:__init(vocabulary,metaSymbols)
	self.errorObservations = {}

	for _,v in ipairs(vocabulary) do
		if not inSet(metaSymbols,v) then
			self.errorObservations[v] = {}
		end
	end
end

function RefWordTranslError:extractError(sentN,positN,refSequenceVector)
	local errorVal = refSequenceVector[positN]
	table.insert(self.errorObservations[errorVal],{sentN,positN})
end

-- v1 is correct, v2 is prediction
local ConfusionTranslError, parent = torch.class('torch.ConfusionTranslError', 'LocalTranslError')

function ConfusionTranslError:__init(vocabulary,metaSymbols)
	self.errorObservations = {}
		for _,v1 in ipairs(vocabulary) do
			if not inSet(metaSymbols,v1) then
				self.errorObservations[v1] = {}
				for _,v2 in ipairs(vocabulary) do
					if not inSet(metaSymbols,v2) then
						self.errorObservations[v1][v2] = {}
					end
				end
			end
		end
end

function ConfusionTranslError:extractError(sentN,positN,refSequenceVector,predSequenceVector)
	local correctVal = refSequenceVector[positN]
	local predVal = predSequenceVector[positN]
	table.insert(self.errorObservations[correctVal][predVal],{sentN,positN})
end


local NgramContextTranslError, parent = torch.class('torch.NgramContextTranslError', 'LocalTranslError')

function NgramContextTranslError:__init(vocabulary,metaSymbols,ngramOrder)
	self.contextLen = ngramOrder 
	self.errorObservations = {}

	for _,v in ipairs(vocabulary) do
		if not inSet(metaSymbols,v1) then
			self.errorObservations[v] = {}
		end
	end
end

function NgramContextTranslError:extractError(sentN,positN,refSequenceVector)
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
	if currTable[-1] == nil then currTable[-1] = {} end
	--print(currTable)
	table.insert(currTable[-1],{sentN,positN})
end
	
	

	


