local layer, parent = torch.class('nn.GRU', 'nn.Module')

function layer:__init(inputSize, hiddenSize)
    parent.__init(self)
    local D, H = inputSize, hiddenSize
    self.inputSize, self.hiddenSize = D, H

    self.weight = torch.Tensor(D + 3 * H, 3 * H)
    self.gradWeight = torch.Tensor(D + 3 * H, 3 * H)
    self.bias = torch.Tensor(6 * H)
    self.gradBias = torch.Tensor(6 * H)
    self:reset()
    self.gates = torch.Tensor()
    self.context = torch.Tensor()
    -- buffer
    self.pattn = torch.Tensor()
    self.buffer_h = torch.Tensor() -- buffer for intermediate h
    self.attn_buffer = torch.Tensor() -- attention buffer
    self.attn_probs = torch.Tensor()
    self.buffer = torch.Tensor() -- (N, T, H) Tensor
    self.grad_next_h = torch.Tensor()
    self.buffer_b = torch.Tensor()
    self.grad_cntx = torch.Tensor()
    self.grad_prime_h = torch.Tensor()
    self.grad_a_buffer = torch.Tensor()
    self.gradInput = torch.Tensor()

    self.h0 = torch.Tensor()
    self.remember_states = false
    self.grad_h0 = torch.Tensor()
end


function layer:reset(std)
    if not std then
        std = 1.0 / math.sqrt(self.hiddenSize + self.inputSize)
    end
    self.bias:zero()
    self.weight:normal(0,std)
    return self
end


function layer:resetStates()
    self.h0 = self.h0.new()
end


function layer:lastStates()
    local prev_T = self.output:size(2)
    return {self.output[{{}, prev_T}]}
end


function layer:setStates(state)
    local h0 = state[1]
    self.h0:resizeAs(h0):copy(h0)
    self._set_state = true
end


local function check_dims(x, dims)
    assert(x:dim() == #dims)
    for i, d in ipairs(dims) do
        assert(x:size(i) == d)
    end
end


function layer:_unpack_input(input)
    local h0, x = nil, nil
    if torch.type(input) == 'table' and #input == 2 then
        h0, x = unpack(input)
    elseif torch.isTensor(input) then
        x = input
    else
        assert(false, 'invalid input')
    end
    return h0, x
end


function layer:_get_sizes(input, gradOutput)
    local h0, x = self:_unpack_input(input)
    local N, T = x:size(1), x:size(2)
    local H, D = self.hiddenSize, self.inputSize
    check_dims(x, {N, T, D})
    if h0 then
        check_dims(h0, {N, H})
    end
    if gradOutput then
        check_dims(gradOutput, {N, T, H})
    end
  return N, T, D, H
end


function layer:setGrad(grad0)
    self._set_grad = true
    self.grad_h0 = grad0[1]
end


function layer:getGrad()
    return {self.grad_h0}
end

function layer:updateOutput(input)
    local x, _m = unpack(input)
    local Tm = _m:size(2)
    local h0 = self.h0
    local N, T = x:size(1), x:size(2)
    local D = self.inputSize
    local H = self.hiddenSize
    self.mem = _m:transpose(2, 3) -- will be (N, H, Tm)
    self._return_grad_h0 = (h0 ~= nil)

    if h0:nElement() == 0 or not self._set_state then
        h0:resize(N, H):zero()
    end


    local Wx1 = self.weight[{{1, D}, {}}]
    local Wh1 = self.weight[{{D + 1, D + H}, {}}]

    local Wx2 = self.weight[{{D + H + 1, D + 2 * H}, {}}]
    local Wh2 = self.weight[{{D + 2 * H + 1, D + 3 * H}, {}}]

    local h = self.output
    h:resize(N, T, H):fill(0)

    self.buffer_h:resize(N, T, H):fill(0)
    local prev_h = h0

    self.gates:resize(2 * N, T, 3 * H):zero()
    self.buffer:resize(2 * N, T, 3 * H):zero()

    -- expand the bias to the batch_size
    local bias_expand = self.bias:view(1, 1, -1):expand(N, T, 6 * H)
    -- pre add bias
    self.gates[{{1, N}}]:add(bias_expand:narrow(3, 1, 3 * H))
    self.gates[{{N + 1, 2 * N}}]:add(bias_expand:narrow(3, 3 * H + 1, 3 * H))

    self.attn_buffer:resize(N, T, Tm)
    self.attn_probs:resize(N, T, Tm)
    self.context:resize(N, T, H)
    for t = 1, T do
        -- compute the first recurrent cell
        local cur_x = x[{{}, t}]
        local cur_buffer = self.buffer[{{1, N}, t}]
        local cur_gates = self.gates[{{1, N}, t}]
        cur_gates:addmm(cur_x, Wx1)
        cur_buffer:mm(prev_h, Wh1)
        local rz  = cur_gates[{{}, {H + 1, 3 * H}}]
        rz:add(cur_buffer[{{}, {H + 1, 3 * H}}])
        rz:sigmoid()

        -- get our reset gate
        local r = rz[{{}, {1, H}}]
        local z = rz[{{}, {H + 1, 2 * H}}]
        local hc = cur_gates[{{}, {1, H}}]:addcmul(cur_buffer[{{}, {1, H}}], r):tanh()

        local prime_h = self.buffer_h[{{}, t}]-- store intermediate hidden
        prime_h:addcmul(hc, -1, z, hc)
        prime_h:addcmul(z, prev_h)

        local attn_scores = self.attn_buffer[{{}, {t}}]
        attn_scores:bmm(self.buffer_h[{{}, {t}}], self.mem)
        local attn_score2d = self.attn_buffer[{{}, t}]

        attn_score2d.THNN.SoftMax_updateOutput(
            attn_score2d:cdata(),
            self.pattn:cdata()
        )
        self.attn_probs[{{}, t}]:copy(self.pattn)

        local attn_probs = self.attn_probs[{{}, {t}}] -- will be (N, 1, Tx)
        local cur_c = self.context[{{}, {t}}] -- will be (N, 1, H)
        cur_c:bmm(attn_probs, _m)

        -- next layer
        local prev_h = prime_h
        local cur_x = self.context[{{}, t}]
        local cur_buffer = self.buffer[{{N + 1, 2 * N}, t}]
        local cur_gates = self.gates[{{N + 1, 2 * N}, t}]
        cur_gates:addmm(cur_x, Wx2)
        cur_buffer:mm(prev_h, Wh2)

        local rz  = cur_gates[{{}, {H + 1, 3 * H}}]
        rz:add(cur_buffer[{{}, {H + 1, 3 * H}}])
        rz:sigmoid()
        local r = rz[{{}, {1, H}}]
        local z = rz[{{}, {H + 1, 2 * H}}]
        local hc = cur_gates[{{}, {1, H}}]:addcmul(cur_buffer[{{}, {1, H}}], r):tanh()
        local next_h = h[{{}, t}]
        next_h:addcmul(hc, -1, z, hc)
        next_h:addcmul(z, prev_h)
        prev_h = next_h
    end
    return self.output
end

function layer:backward(input, gradOutput, scale)
    scale = scale or 1.0
    local h0 = self.h0
    local x, _m = unpack(input)
    local grad_h0, grad_x = self.grad_h0, self.gradInput
    local h = self.output
    local grad_h = gradOutput

    local Wx1 = self.weight[{{1, D}, {}}]
    local Wh1 = self.weight[{{D + 1, D + H}, {}}]

    local Wx2 = self.weight[{{D + H + 1, D + 2 * H}, {}}]
    local Wh2 = self.weight[{{D + 2 * H + 1, D + 3 * H}, {}}]

    local grad_Wx1 = self.gradWeight[{{1, D}, {}}]
    local grad_Wh1 = self.gradWeight[{{D + 1, D + H}, {}}]
    local grad_Wx2 = self.gradWeight[{{D + H + 1, D + 2 * H}, {}}]
    local grad_Wh2 = self.gradWeight[{{D + 2 * H + 1, D + 3 * H}, {}}]

    local grad_b = self.gradBias

    if not self._set_grad then
        grad_h0:resizeAs(h0):zero()
    end

    grad_x:resizeAs(x):zero()
    local grad_next_h = self.grad_next_h:resizeAs(h0):copy(grad_h0)

    for t = T, 1, -1 do
        -- backprop all the way down
        local next_h = h[{{}, t}]
        local cur_buffer = self.buffer[{{N + 1, 2 * N}, t}]
        local cur_gates = self.gates[{{N + 1, 2 * N}, t}]
        local prev_h = nil
        if t == 1 then
            prev_h = h0
        else
            prev_h = h[{{}, t - 1}]
        end

        grad_next_h:add(grad_h[{{}, t}])

        local hc = cur_gates[{{}, {1, H}}]
        local z = cur_gates[{{}, {H + 1, 2 * H}}]
        local r = cur_gates[{{}, {2 * H + 1, 3 * H}}]

        -- fill with 1 for convenience
        local grad_a = self.grad_a_buffer:resize(N, 3 * H):zero()
        local grad_az = grad_a[{{}, {H + 1, 2 * H}}]
        local grad_ar = grad_a[{{}, {2 * H + 1, 3 * H}}]
        local grad_ah = grad_a[{{}, {1, H}}]

        local prime_h = self.buffer_h[{{}, t}]
        -- use grad_ar as buffer
        grad_ar:add(prime_h, -1, hc):cmul(grad_next_h)
        grad_az:addcmul(z, -1, z, z):cmul(grad_ar)
        -- use grad_ar to store grad comes to hc
        grad_ar:zero():addcmul(grad_next_h, -1, z, grad_next_h)
        -- derivative of tanh, scale by grad_ar
        grad_ah:addcmul(-1, hc, hc):cmul(grad_ar)
        grad_ar:zero():addcmul(r, -1, r, r):cmul(grad_ah):cmul(cur_buffer[{{}, {1, H}}])

        -- we have enough info to compute grad_Wx2
        local cur_x = self.context[{{}, t}]
        grad_Wx2:addmm(scale, cur_x:t(), grad_a)
        -- grad to context is easy
        local grad_cntx = self.grad_cntx
        local grad_ph = self.grad_prime_h
        grad_cntx:mm(grad_a, Wx2:t())
        -- ok
        grad_ah:cmul(r)
        grad_Wh2:addmm(scale, prime_h:t(), grad_a)
        -- now compute grad to context and prime_h
        grad_ph:mm(grad_a, Wh2:t())
        grad_ph:addcmul(grad_ah, z)


        local grad_attn = torch.Tensor(N, 1, Tm)
        local attn_probs = self.attn_probs[{{}, {t}}]
        grad_attn:bmm(grad_cntx:view(N, 1, H), self.mem)

        -- TODO: passing to softmax and down to the lower layer
        -- accumulate from reset and update gate
    end

    grad_h0:copy(grad_next_h)

    return self.gradInput
end

function layer:updateGradInput(input, gradOutput)
    self:backward(input, gradOutput, 0)
end

function layer:accGradParameters(input, gradOutput, scale)
    self:backward(input, gradOutput, scale)
end

function layer:clearState()
    nn.utils.clear(self, {
        'output',
        'gates',
        'buffer_b',
        'grad_h0',
        'grad_next_h',
        'grad_a_buffer',
        'gradInput'
    })
end
