--require 'localTranslError.RefWordTranslError' 
--require 'localTranslError.ConfusionTranslError'
--require 'localTranslError.NgramContextTranslError'
require 'localTranslError'

--util function

function inputVocabulary()
	return {1,2,3,4,5,6,7,8,9,10}, {} -- {1,2,3}
end

function example1()
	local predBatch1 = torch.Tensor({{1,2,4},{3,4,5}})
	local refBatch1 = torch.Tensor({{3,2,4},{3,3,5}})
	local error1 = torch.Tensor({{1,0,0},{0,1,0}})
	return predBatch1,refBatch1,error1
end


function example2()
	local predBatch2 = torch.Tensor({{4,4,6,7,1},{5,5,2,1,8},{2,5,1,2,3}})
	local refBatch2 = torch.Tensor({{4,2,5,7,2},{5,5,2,1,1},{3,5,1,5,5}})
	local error2 = torch.Tensor({{0,1,1,0,1},{0,0,0,0,1},{1,0,0,1,1}})
	return predBatch2,refBatch2,error2
end

function errorExtractionWrapper(predBatch1,refBatch1,error1,extractor)
	local batchDim1 = predBatch1:size()
	for i=1,batchDim1[1] do
		local refVector = refBatch1[i]
		local predVector = predBatch1[i]
		for j=1,batchDim1[2] do
			if error1[i][j] == 1 then
				extractor:extractError(i,j,refVector,predVector)
			end
		end
	end
	return extractor
end

--tests
local tests = torch.TestSuite()
local tester = torch.Tester()

function tests.test_inSet()
	local set1 = {1,2,3}
	tester:assert(inSet(set1,2))
	tester:assert(not inSet(set1,6))

	local set2 = {4,3,2,1}
	tester:assert(inSet(set2,3))
	tester:assert(not inSet(set2,0))
end


function tests.test_errorIdentification_1()	
	local predBatch1,refBatch1,error1 = example1()
	local pred_er1 = torch.ne(predBatch1,refBatch1)
	tester:eq(pred_er1,error1:byte())
end


function tests.test_errorIdentification_2()
	local predBatch2,refBatch2,error2 = example2()
	local pred_er2 = torch.ne(predBatch2,refBatch2)
	tester:eq(pred_er2,error2:byte())
end



function tests.testRefWordTranslError_1()
	local vocab, metasymbols = inputVocabulary()
	local errorEtractor = torch.RefWordTranslError(vocab,metasymbols)
	local predBatch1,refBatch1,error1 = example1()

	errorExtractionWrapper(predBatch1,refBatch1,error1,errorEtractor)

	tester:assertGeneralEq(errorEtractor.errorObservations[1],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[2],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[3],{{1,1},{2,2}})
	tester:assertGeneralEq(errorEtractor.errorObservations[4],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[5],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[6],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[7],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[8],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[9],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[10],{})

end

function tests.testRefWordTranslError_2()
	local vocab, metasymbols = inputVocabulary()
	local errorEtractor = torch.RefWordTranslError(vocab,metasymbols)
	local predBatch2,refBatch2,error2 = example2()

	errorExtractionWrapper(predBatch2,refBatch2,error2,errorEtractor)

	tester:assertGeneralEq(errorEtractor.errorObservations[1],{{2,5}})
	tester:assertGeneralEq(errorEtractor.errorObservations[2],{{1,2},{1,5}})
	tester:assertGeneralEq(errorEtractor.errorObservations[3],{{3,1}})
	tester:assertGeneralEq(errorEtractor.errorObservations[4],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[5],{{1,3},{3,4},{3,5}})
	tester:assertGeneralEq(errorEtractor.errorObservations[6],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[7],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[8],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[9],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[10],{})
end

function tests.test_ConfusionTranslError_1()
	local vocab, metasymbols = inputVocabulary()
	local predBatch1,refBatch1,error1 = example1()
	local errorEtractor = torch.ConfusionTranslError(vocab,metasymbols)

	errorExtractionWrapper(predBatch1,refBatch1,error1,errorEtractor)

	tester:assertGeneralEq(errorEtractor.errorObservations[3][1],{{1,1}})
	tester:assertGeneralEq(errorEtractor.errorObservations[3][4],{{2,2}})
	tester:assertGeneralEq(errorEtractor.errorObservations[3][2],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[3][5],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[1][3],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[1][1],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[4][3],{})
end

function tests.test_ConfusionTranslError_2()
	local vocab, metasymbols = inputVocabulary()
	local predBatch2,refBatch2,error2 = example2()
	local errorEtractor = torch.ConfusionTranslError(vocab,metasymbols)

	errorExtractionWrapper(predBatch2,refBatch2,error2,errorEtractor)

	tester:assertGeneralEq(errorEtractor.errorObservations[2][1],{{1,5}})
	tester:assertGeneralEq(errorEtractor.errorObservations[2][4],{{1,2}})
	tester:assertGeneralEq(errorEtractor.errorObservations[4][2],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[1][2],{})
	tester:assertGeneralEq(errorEtractor.errorObservations[5][6],{{1,3}})
	tester:assertGeneralEq(errorEtractor.errorObservations[5][2],{{3,4}})
	tester:assertGeneralEq(errorEtractor.errorObservations[5][3],{{3,5}})	
	tester:assertGeneralEq(errorEtractor.errorObservations[1][8],{{2,5}})
	tester:assertGeneralEq(errorEtractor.errorObservations[3][2],{{3,1}})
	tester:assertGeneralEq(errorEtractor.errorObservations[4][3],{})
end


function tests.test_2NgramContextTranslError_1()
	local vocab, metasymbols = inputVocabulary()
	local predBatch1,refBatch1,error1 = example1()
	local errorEtractor = torch.NgramContextTranslError(vocab,metasymbols,2)

	errorExtractionWrapper(predBatch1,refBatch1,error1,errorEtractor)

	tester:assertGeneralEq(errorEtractor.errorObservations[3][-1],{{1,1}})
	tester:assertGeneralEq(errorEtractor.errorObservations[3][3][-1],{{2,2}})
	
end


function tests.test_2NgramContextTranslError_2()
	local vocab, metasymbols = inputVocabulary()
	local predBatch2,refBatch2,error2 = example2()
	local errorEtractor = torch.NgramContextTranslError(vocab,metasymbols,2)

	errorExtractionWrapper(predBatch2,refBatch2,error2,errorEtractor)
	
	tester:assertGeneralEq(errorEtractor.errorObservations[2][4][-1],{{1,2}})
	tester:assertGeneralEq(errorEtractor.errorObservations[2][7][-1],{{1,5}})
	tester:assertGeneralEq(errorEtractor.errorObservations[5][2][-1],{{1,3}})
	tester:assertGeneralEq(errorEtractor.errorObservations[5][1][-1],{{3,4}})
	tester:assertGeneralEq(errorEtractor.errorObservations[5][5][-1],{{3,5}})
	tester:assertGeneralEq(errorEtractor.errorObservations[1][1][-1],{{2,5}})
	tester:assertGeneralEq(errorEtractor.errorObservations[3][-1],{{3,1}})
	
end

tester:add(tests)
tester:run()