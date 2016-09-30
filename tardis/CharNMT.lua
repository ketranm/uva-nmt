-- author: Ke Tran <m.k.tran@uva.nl>

require 'tardis.GlimpseDot'
require 'optim'
require 'tardis.SeqAtt'
require 'tardis.FastTransducer'
require 'tardis.CharTransducer'
local model_utils = require 'tardis.model_utils'
local CharNMT, parent = torch.class('nn.CharNMT', 'nn.NMT')

function CharNMT:__init(opt)
    -- build encoder
    local sourceSize = opt.sourceSize
    local inputSize = opt.inputSize
    local hiddenSize = opt.hiddenSize
    self.encoder = nn.CharTransducer(opt)

    -- build decoder
    local targetSize = opt.targetSize
    self.decoder = nn.Transducer(targetSize, inputSize, hiddenSize, opt.numLayers, opt.dropout)

    -- attention
    self.glimpse = nn.GlimpseDot()

    self.layer = nn.Sequential()
    self.layer:add(nn.JoinTable(3))
    self.layer:add(nn.View(-1, 2 * hiddenSize))
    self.layer:add(nn.Linear(2 * hiddenSize, hiddenSize, false))
    self.layer:add(nn.Tanh())
    self.layer:add(nn.Linear(hiddenSize, targetSize, true))
    self.layer:add(nn.LogSoftMax())

    local weights = torch.ones(targetSize)
    weights[opt.padIdx] = 0

    self.padIdx = opt.padIdx
    self.criterion = nn.ClassNLLCriterion(weights, true)
    -- convert to cuda
    self.encoder:cuda()
    self.decoder:cuda()
    self.glimpse:cuda()
    self.layer:cuda()
    self.criterion:cuda()

    self.params, self.gradParams =
        model_utils.combine_all_parameters(self.encoder,
                                           self.decoder,
                                           self.glimpse,
                                           self.layer)
    self.maxNorm = opt.maxNorm or 5
    -- for optim
    self.optimConfig = {}
    self.optimStates = {}
end
