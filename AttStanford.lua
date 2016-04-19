require 'nngraph'
require 'PrintIdentity'

local Att = {}
function Att.attention(rnn_size, project_size, conv_feature_size, num_conv_feature)

    local prev_h = nn.Identity()()
    local C = nn.Identity()()

    -- compute attention coefficients
    local flatten_conv = nn.View(-1):setNumInputDims(2)(C)
    local f_conv = nn.Linear(conv_feature_size*num_conv_feature, num_conv_feature)(flatten_conv)
    local f_h = nn.Linear(rnn_size, num_conv_feature)(prev_h)
    local f_sum = nn.Tanh()(nn.CAddTable()({f_conv, f_h}))
    local coef = nn.SoftMax()(f_sum)
    local coef_expanded = nn.Reshape(num_conv_feature, 1)(coef)
    -- compute soft spatial attention
    local soft_att = nn.MM(true, false)({coef_expanded, C})
    local att_conv = nn.View(-1):setNumInputDims(2)(soft_att)
    local att_out = nn.ReLU()(nn.Linear(conv_feature_size, rnn_size)(att_conv))

    outputs = {}
    table.insert(outputs, att_out)

    -- packs the graph into a convenient module with standard API (:forward(), :backward())
    return nn.gModule({prev_h, C}, outputs)
end

return Att
