require 'data.AbstractDataLoader'
local _ = require 'moses'

local MultiDataLoader, parent = torch.class('MultiDataLoader', 'AbstractDataLoader')

function MultiDataLoader:__init(mainKwargs,multiKwargs)
    parent.__init(self)
    self.dataPath = mainKwargs.dataPath
	--data
    self.langs = {trg = mainKwargs.target}
    for i,kwargs in ipairs(multiKwargs) do self.langs[i] = kwargs.source end

    local trainPrefix = mainKwargs.combinTrainPrefix
    local validPrefix = mainKwargs.combinValidPrefix
    local testPrefix = mainKwargs.testPrefix
    local srcLangs = {}
    for langN=1,#self.langs do
	table.insert(srcLangs,self.langs[langN])
    end
    local multiSrcFiles = _.map(srcLangs, function(i,v) return path.join(multiKwargs[i].dataPath,trainPrefix..'.'..v) end)
    --local multiSrcFilesValid = _.map(srcLangs,function(i,v) return path.join(multiKwargs[i].dataPath,validPrefix..'.'..v) end)
    --local multiSrcFilesTest = _.map(srcLangs,function(i,v) return path.join(multiKwargs[i].dataPath,testPrefix..'.'..v) end)
    local trgFile = path.join(multiKwargs[1].dataPath,trainPrefix..'.'..mainKwargs.target)
    --local trgFileValid = path.join(multiKwargs[1].dataPath,validPrefix..'.'..mainKwargs.target)
    --local trgFileTest = path.join(multiKwargs[1].dataPath,testPrefix..'.'..mainKwargs.target)

    -- auxiliary file to store additional information about shards
    local indexfile = path.join(mainKwargs.dataPath, 'index.t7')
    self.trgVocabSize = mainKwargs.trgVocabSize
	self.srcVocabSizes = _.map(multiKwargs,function(i,v) return v.srcVocabSize end)
	self.vocabs = _.map(multiKwargs, function(i,kw) 
			local vocFile =path.join(kw.dataPath,'vocab.t7') 
			local voc = torch.load(vocFile)
			return voc
			end)
	if not path.exists(indexfile) then 
	local batchSize = mainKwargs.batchSize
    	local shardSize = mainKwargs.shardSize
    	 print('=> create multisource training tensor files...')
		self:text2tensor(multiSrcFiles,trgFile,shardSize,batchSize,self.tracker[1])
		print('=> create multisource validation tensor files...')
		--self:text2tensor(multiSrcFilesValid,trgFileValid,shardSize,batchSize,self.tracker[2])
		torch.save(indexfile, self.tracker)
    else
        self.tracker = torch.load(indexfile)
    end
    self.padIdx = self.vocabs[1][2].idx('<pad>')
    assert(self.padIdx == 1)
end

function multiFileIterator(multiFiles)
	local fileIterators = _.map(multiFiles,function(i,file) return io.lines(file) end)
	local numLines = 0
    for _ in io.lines(multiFiles[1]) do numLines = numLines+ 1 end
   	local i = 0
   	local iter = function () 
        i = i+1
        if i <= numLines then return _.map(fileIterators,function(k,v) return v() end) end
    end
    return iter
end

function MultiDataLoader:text2tensor(srcFiles, trgFile, shardSize, batchSize, tracker)
    --[[Load source and target text file and save to tensor format.
    If the files are too large, process a shard of shardSize sentences at a time
    --]]

   local trgIterator = io.lines(trgFile)
   local multiSrcIterator = multiFileIterator(srcFiles)
    -- helper
    local nsents = 0 -- sentence counter
    local buckets = {}
    local diff = 1 -- maximum different in length of the target
    local prime = 997 -- use a large prime number to avoid hash collision

    while true do
    	local multiSrc =  multiSrcIterator()
        if multiSrc == nil then break end
        local trg = trgIterator()
        local multiSrcTokens = _.map(multiSrc,function(i,v) return  stringx.split(v) end)
        local trgTokens = stringx.split(trg)
        if _.reduce(multiSrcTokens,function(memo,v) return memo and (#v ~=0) end) and  #trgTokens ~= 0 then --omit empty sentences
        	nsents = nsents + 1
        	local srcIdx = _.map(multiSrc, function(i,v) return self.encodeString(v,self.vocabs[i][1],'none') end)
        	local trgIdx = self.encodeString(trg, self.vocabs[1][2], 'both')  -- trg.vocabularyes the same across lang.pairs
        	local trgLen = #trgIdx + diff - (#trgIdx % diff)
        	-- hashing
	        local bidx = trgLen
	        for i=1,#srcIdx do bidx = bidx*prime + #srcIdx[i] end
	        -- reverse the source sentence
	        srcIdx = _.map(srcIdx, function(i,v) return _.reverse(v) end)
			for i = 1, trgLen - #trgIdx do table.insert(trgIdx, self.vocabs[1][2].idx('<pad>')) end
			-- put sentence pairs to corresponding bucket
	        buckets[bidx] = buckets[bidx] or {_.map(srcFiles, function(i,v) return {} end), {}}
    	    local bucket = buckets[bidx]
	    for i,b in ipairs(bucket[1]) do
		table.insert(bucket[1][i],srcIdx[i])
	    end
            table.insert(bucket[2], trgIdx)
            if  nsents % shardSize == 0 then
            	self:saveShard(buckets, batchSize, tracker)
            	buckets = {}
            end
           	if nsents % shardSize  == 0 then
        		self:saveShard(buckets, batchSize, tracker)
    		end
        end

    end	       
    if nsents % shardSize  > 1 then
        self:saveShard(buckets, batchSize, tracker)
    end
end


function MultiDataLoader:saveShard(buckets, batchSize, tracker)
    local shard = {}
    for bidx, bucket in pairs(buckets) do
        -- make a big torch.IntTensor matrix
        local ss = _.map(bucket[1], function(i,v) return torch.IntTensor(v):split(batchSize, 1) end)
        local ts = torch.IntTensor(bucket[2]):split(batchSize, 1)
        buckets[bidx] = nil -- free memory
        -- sanity check
        assert(#ss[1] == #ts)
        assert(#ss[2] == #ss[2])
        for i = 1, #ts do
            table.insert(shard, { _.map(ss, function(k,v) return v[i] end), ts[i]})
        end
	
    end

    if not tracker.fidx then tracker.fidx = 0 end
    tracker.fidx = tracker.fidx + 1
    print("Saving to ", string.format('%s/%s.shard_%d.t7',
                                self.dataPath, tracker.name, tracker.fidx))
    local file = string.format('%s/%s.shard_%d.t7',
                                self.dataPath, tracker.name, tracker.fidx)
    torch.save(file, shard)
    tracker.size = (tracker.size or 0) + #shard
    collectgarbage()
end
