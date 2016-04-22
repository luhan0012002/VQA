require 'nn'
require 'rnn'
require 'getData'
require 'optim'
require 'cutorch'
require 'cunn'
require 'PrintIdentity'
require 'FastLSTM_padding'

local LSTM = require 'LSTM'

local model = {}
function model.buildModel(attMethod)
    local Att
    if attMethod == 'ReLU' then
        Att = require 'AttReLU'
    elseif attMethod == 'noActivation' then
        Att = require 'Att'
    elseif attMethod == 'AttStanford' then
        Att = require 'AttStanford'
    else print('wrong attention model!')
    end

    local protos = {}
    protos.lstm = LSTM.lstm(hiddenSize, hiddenSize, convFeatureSize):cuda()
    --protos.lstm:maskZero(1)
    protos.attention = Att.attention(hiddenSize, projectSize, convFeatureSize, numConvFeature):cuda()
    protos.wordEmbed = nn.LookupTableMaskZero(nIndex, hiddenSize):cuda()
    protos.imageEmbed = nn.Linear(fcSize, hiddenSize):cuda()
    protos.classify = nn.Sequential():add(nn.Linear(hiddenSize, nClass)):add(nn.LogSoftMax()):cuda()
    protos.criterion = nn.ClassNLLCriterion():cuda()

    --protos.lstm, protos.wordEmbed, protos.imageEmbed, protos.classify, protos.criterion = protos.lstm:cuda(), protos.wordEmbed:cuda(), protos.imageEmbed:cuda(), protos.classify:cuda, protos.criterion:cuda()

    return protos
end

return model

