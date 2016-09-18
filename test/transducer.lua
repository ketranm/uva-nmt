require 'nn'
require 'tardis.Transducer'


local gradcheck = require 'misc.gradcheck'
local tests = torch.TestSuite()
local tester = torch.Tester()


local function check_size(x, dims)
    tester:assert(x:dim() == #dims)
        for i, d in ipairs(dims) do
            tester:assert(x:size(i) == d)
        end
end

vocabSize = 10000
inputSize = 4
hiddenSize = 4
numLayers = 3


function tests.forward_backward()
    -- generate example

    local N = torch.random(20, 200)
    local T = torch.random(10, 50)
    local x = torch.range(1, N * T):reshape(N, T)

    --print(vocabSize, inputSize, hiddenSize, numLayers)
    local transducer = nn.Transducer(vocabSize, inputSize, hiddenSize, numLayers)

    -- test number of parameter
    local D, H = inputSize, hiddenSize

    local num_params = (H + D + 1) * 4 * H + (numLayers - 1) * (2 * H + 1) * (4 * H) + vocabSize * D

    local params, grad_params = transducer:getParameters()
    tester:assert(params:nElement() == num_params)
    tester:assert(grad_params:nElement() == num_params)

    local lt = nn.LookupTable(vocabSize, inputSize)
    lt.weight:copy(transducer.net:get(1).weight)
    local rnns = {}

    local initStates = {}

    for i = 1, numLayers do
        local c0 = torch.randn(N, hiddenSize)
        local h0 = torch.randn(N, hiddenSize)
        initStates[i] = {c0, h0}
    end

    for i = 1, numLayers do
        local prev_h = hiddenSize
        if i == 1 then prev_h = inputSize end
        local rnn = nn.LSTM(prev_h, hiddenSize)
        rnn.weight:copy(transducer._rnns[i].weight)  -- reset weight
        rnn.bias:copy(transducer._rnns[i].bias)
        rnn:setStates(initStates[i])
        table.insert(rnns, rnn)
    end
    -- set state of transducer
    transducer:setStates(initStates)

    local wemb = lt:forward(x)  -- word embeddings
    local h = wemb
    local hx = {[0] = h}
    for i = 1, numLayers do
        local h_next = rnns[i]:forward(h)
        h = h_next
        hx[i] = h
    end

    local h_trans = transducer:forward(x)
    tester:assertTensorEq(h, h_trans, 1e-10)

    -- test backward
    local grad = torch.Tensor():resizeAs(h_trans):uniform(0,1):mul(0.1)
    transducer:backward(x, grad)
    local prev_grad
    for i = numLayers, 1, -1 do
        if i == numLayers then
            prev_grad = grad
        end
        local grad_i = rnns[i]:backward(hx[i-1], prev_grad)
        prev_grad = grad_i
    end
    lt:backward(x, prev_grad)

    tester:assertTensorEq(transducer.net:get(1).gradWeight, lt.gradWeight, 1e-10)

    for i = 1, numLayers do
        tester:assertTensorEq(transducer._rnns[i].gradWeight, rnns[i].gradWeight, 1e-10)
        tester:assertTensorEq(transducer._rnns[i].gradBias, rnns[i].gradBias, 1e-10)
    end
end


function tests.gradcheck()
    -- generate example

    local N = 2
    local T = 3
    local x = torch.range(1, N * T):reshape(N, T)


    local transducer = nn.Transducer(vocabSize, inputSize, hiddenSize, numLayers)

    local state0 = {}

    for i = 1, numLayers do
        local c0, h0
        c0 = torch.randn(N, hiddenSize)
        h0 = torch.randn(N, hiddenSize)
        state0[i] = {c0, h0}
    end

    transducer:setStates(state0)


    local h = transducer:forward(x)
    local grad = torch.randn(#h)
    transducer:backward(x, grad)

    local function fh0(h0)
        state0[3][2] = h0
        transducer:setStates(state0)
        return transducer:forward(x)
    end

    local function fc0(c0)
        state0[3][1] = c0
        transducer:setStates(state0)
        return transducer:forward(x)
    end

    local c0, h0 = unpack(state0[3])
    local grad_state = transducer:gradStates()
    local dc0, dh0 = unpack(grad_state[3])

    local dh0_num = gradcheck.numeric_gradient(fh0, h0, grad, 1e-12)
    local dc0_num = gradcheck.numeric_gradient(fc0, c0, grad, 1e-12)

    local dh0_error = gradcheck.relative_error(dh0_num, dh0)
    local dc0_error = gradcheck.relative_error(dc0_num, dc0)

    tester:assertle(dh0_error, 1e-2, "gradcheck")
    tester:assertle(dc0_error, 1e-2, "gradcheck")
end

tester:add(tests)
tester:run()
