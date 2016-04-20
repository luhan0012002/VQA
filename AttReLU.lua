require 'nngraph'
require 'PrintIdentity'

local Att = {}
function Att.attention(rnn_size, project_size, conv_feature_size, num_conv_feature)
    local prev_h = nn.Identity()()
    local C = nn.Identity()()

    local C_reshape = nn.View(conv_feature_size):setNumInputDims(1)(C)
    local C2h = nn.Linear(conv_feature_size, project_size)(C_reshape)      -- C_reshape to hidden (batchSize*196, 512)
    local C2h_reshape = nn.View(num_conv_feature, project_size):setNumInputDims(2)(C2h)
    local h2h = nn.Linear(rnn_size, project_size)(prev_h)           -- hidden to hidden (batchSize, 512)
    local h_repeat = nn.Replicate(num_conv_feature , 2, 2)(h2h)   -- repeat h (batchSize*196, 512)
    local preactivations = nn.CAddTable()({C2h_reshape, h_repeat})
    local preactivations_reshape = nn.View(project_size):setNumInputDims(1)(preactivations)
    local activations = nn.Tanh()(preactivations_reshape)
    local e = nn.Linear(project_size, 1)(activations)
    local e_reshape = nn.View(num_conv_feature):setNumInputDims(2)(e)
    local a = nn.SoftMax()(e_reshape)
    local a_reshape = nn.View(1,3):setNumInputDims(2)(a)
    local r = nn.MM(false, false)({a, C})
    local r_reshape = nn.Reshape(conv_feature_size)(r) 
    local r_activate = nn.ReLU(true)(r_reshape)
    outputs = {}
    table.insert(outputs, r_activate)
    -- packs the graph into a convenient module with standard API (:forward(), :backward())
    return nn.gModule({prev_h, C}, outputs)
end

return Att
