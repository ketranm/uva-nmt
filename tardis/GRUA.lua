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
    self.gradInput = {torch.Tensor(), torch.Tensor()}

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

function layer:updateOutput(input)
    local x, _m = unpack(input)
    local M = _m:size(2)
    local h0 = self.h0
    local N, T = x:size(1), x:size(2)
    local D = self.inputSize
    local H = self.hiddenSize
    self.mem = _m:transpose(2, 3) -- will be (N, H, M)
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

    self.attn_buffer:resize(N, T, M)
    self.attn_probs:resize(N, T, M)
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

        local attn_input = self.attn_buffer[{{}, {t}}]
        attn_input:bmm(self.buffer_h[{{}, {t}}], self.mem)
        local attn_input = self.attn_buffer[{{}, t}]

        attn_input.THNN.SoftMax_updateOutput(
            attn_input:cdata(),
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
    local x, _m = unpack(input)
    local M = _m:size(2)
    -- nicely alias
    local N, T = x:size(1), x:size(2)
    local D = self.inputSize
    local H = self.hiddenSize
    local h0 = self.h0

    local grad_h0 = self.grad_h0
    local grad_x, grad_m = self.gradInput[1], self.gradInput[2]
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


        grad_next_h:add(grad_h[{{}, t}])

        local hc = cur_gates[{{}, {1, H}}]
        local z = cur_gates[{{}, {H + 1, 2 * H}}]
        local r = cur_gates[{{}, {2 * H + 1, 3 * H}}]

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
        self.grad_cntx:resize(N, H):mm(grad_a, Wx2:t())
        grad_ah:cmul(r)
        grad_Wh2:addmm(scale, prime_h:t(), grad_a)
        self.grad_prime_h:resize(N, H):mm(grad_a, Wh2:t()):addcmul(grad_ah, z)
        -- ok, we do not need grad_a now
        -- maybe reuse it?
        local grad_pattn = self.grad_a_buffer:resize(N, 1, M)
        grad_pattn:bmm(self.grad_cntx:view(N, 1, H), self.mem)
        local cur_pattn = self.attn_probs[{{}, t}]
        local grad_pattn = self.grad_a_buffer[{{}, 1}]
        self.pattn:copy(cur_pattn)

        local attn_input = self.attn_buffer[{{}, t}]
        local grad_attn_inp = self.buffer_b
        attn_input.THNN.SoftMax_updateGradInput(
            attn_input:cdata(),
            grad_pattn:cdata(),
            grad_attn_inp:cdata(),
            self.pattn:cdata()
        )
        -- use grad_h0
        local da_dh = self.grad_h0:resize(N, 1, H)
        da_dh:bmm(grad_attn_inp:view(N, 1, M), _m)
        self.grad_prime_h:add(da_dh)


        local cur_buffer = self.buffer[{{1, N}, t}]
        local cur_gates = self.gates[{{1, N}, t}]

        local hc = cur_gates[{{}, {1, H}}]
        local z = cur_gates[{{}, {H + 1, 2 * H}}]
        local r = cur_gates[{{}, {2 * H + 1, 3 * H}}]

        local grad_a = self.grad_a_buffer:resize(N, 3 * H):zero()
        local grad_az = grad_a[{{}, {H + 1, 2 * H}}]
        local grad_ar = grad_a[{{}, {2 * H + 1, 3 * H}}]
        local grad_ah = grad_a[{{}, {1, H}}]

        local prev_h = nil
        if t == 1 then
            prev_h = h0
        else
            prev_h = h[{{}, t - 1}]
        end
        local grad_next_h = self.grad_prime_h

        grad_ar:add(prev_h, -1, hc):cmul(grad_next_h)
        grad_az:addcmul(z, -1, z, z):cmul(grad_ar)
        -- use grad_ar to store grad comes to hc
        grad_ar:zero():addcmul(grad_next_h, -1, z, grad_next_h)
        -- derivative of tanh, scale by grad_ar
        grad_ah:addcmul(-1, hc, hc):cmul(grad_ar)
        grad_ar:zero():addcmul(r, -1, r, r):cmul(grad_ah):cmul(cur_buffer[{{}, {1, H}}])

        local cur_x = x[{{}, t}]
        grad_Wx1:addmm(scale, cur_x:t(), grad_a)
        grad_x[{{}, t}]:mm(grad_a, Wx1:t())

        grad_ah:cmul(r)
        grad_Wh1:addmm(scale, prev_h:t(), grad_a)
        grad_next_h:mm(grad_a, Wh1:t())
        -- TODO: gradBias and grad mem
        -- now need grad go back to prev_h
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
