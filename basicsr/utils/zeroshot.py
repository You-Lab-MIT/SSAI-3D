import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import torch.autograd as autograd

def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e,sh in zip(elements, shapes):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device))
        return ret_grads
    if type(elements[0]) == list:
        outer = []
        for e,sh in zip(elements, shapes):
            outer.append(broadcast_val(e,sh))
        return outer
    else:
        return broadcast_val(elements, shapes)

def get_layer_metric_array(net, metric, mode): 
    metric_array = []

    for layer in net.modules():
        if mode=='channel' and hasattr(layer,'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
    
    return metric_array

def get_grad_norm_arr(net, inputs, targets, loss_fn, split_data=1, skip_grad=False):
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
        # import ipdb; ipdb.set_trace()
        inputs = inputs[st:en]
        # {k: v[st:en].cuda() for k, v in inputs.items()}
        # inputs[st:en]
        # bert
        # outputs = net.forward(**inputs).logits.cuda()
        # .logits
        outputs = net.forward(inputs[st:en])
        # targets = targets.type(torch.LongTensor).cuda()
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

        grad_norm_arr = get_layer_metric_array(net, lambda l: l.weight.grad.norm() if l.weight.grad is not None else torch.zeros_like(l.weight), mode='param')

    return grad_norm_arr    

def fisher_forward_conv2d(self, x):
    x = F.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
    #intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    return self.act

def fisher_forward_linear(self, x):
    x = F.linear(x, self.weight, self.bias)
    self.act = self.dummy(x)
    return self.act

def compute_fisher_per_weight(net, inputs, targets, loss_fn, mode='channel', split_data=1):
    
    device = inputs.device

    if mode == 'param':
        raise ValueError('Fisher pruning does not support parameter pruning.')

    net.train()
    all_hooks = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            #variables/op needed for fisher computation
            layer.fisher = None
            layer.act = 0.
            layer.dummy = nn.Identity()

            #replace forward method of conv/linear
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(fisher_forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(fisher_forward_linear, layer)

            #function to call during backward pass (hooked on identity op at output of layer)
            def hook_factory(layer):
                def hook(module, grad_input, grad_output):
                    act = layer.act.detach()
                    grad = grad_output[0].detach()
                    if len(act.shape) > 2:
                        g_nk = torch.sum((act * grad), list(range(2,len(act.shape))))
                    else:
                        g_nk = act * grad
                    del_k = g_nk.pow(2).mean(0).mul(0.5)
                    if layer.fisher is None:
                        layer.fisher = del_k
                    else:
                        layer.fisher += del_k
                    del layer.act #without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555
                return hook

            #register backward hook on identity fcn to compute fisher info
            layer.dummy.register_backward_hook(hook_factory(layer))

    N = inputs.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        net.zero_grad()
        outputs = net(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # retrieve fisher info
    def fisher(layer):
        if layer.fisher is not None:
            return torch.abs(layer.fisher.detach())
        else:
            return torch.zeros(layer.weight.shape[0]) #size=ch

    grads_abs_ch = get_layer_metric_array(net, fisher, mode)
    shapes = get_layer_metric_array(net, lambda l : l.weight.shape[1:], mode)

    grads_abs = reshape_elements(grads_abs_ch, shapes, device)

    return grads_abs

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

def compute_snip_per_weight(net, inputs, targets, loss_fn,  mode = 'param', split_data=1):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    N = inputs.shape[0]
    # ['input_ids']
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
        inputs =inputs[st:en]
        # {k: v[st:en].cuda() for k, v in inputs.items()}
        # inputs[st:en]
        outputs = net.forward(inputs)
        # .logits
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()
        # outputs = net.forward(inputs[st:en])
        # # .logits
        # loss = loss_fn(outputs, targets[st:en])
        # loss.backward()

    # select the gradients that we want to use for search/prune
    def snip(layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad)
        else:
            return torch.zeros_like(layer.weight)
    
    grads_abs = get_layer_metric_array(net, snip, mode)
    return grads_abs

def compute_grasp_per_weight(net, inputs, targets, loss_fn, mode = 'param',  T=1, num_iters=1, split_data=1):
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
            layer.weight.requires_grad_(True) # TODO isn't this already true?

    # NOTE original code had some input/target splitting into 2
    # I am guessing this was because of GPU mem limit
    net.zero_grad()
    N = inputs.shape[0]
    # ['input_ids'].
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        #forward/grad pass #1
        grad_w = None
        for _ in range(num_iters):
            #TODO get new data, otherwise num_iters is useless!
            # inputs = inputs[st:en]
            # {k: v[st:en].cuda() for k, v in inputs.items()}
            # inputs[st:en]
            outputs = net.forward(inputs[st:en])/T
            # .logits
            loss = loss_fn(outputs, targets[st:en])
            # loss.backward()
            
            # outputs = net.forward(inputs[st:en])/T
            # # .logits
            # loss = loss_fn(outputs, targets[st:en])
            grad_w_p = autograd.grad(loss, weights, allow_unused=True)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

    
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
        # inputs = inputs[st:en]
        outputs = net.forward(inputs[st:en])/T
        loss = loss_fn(outputs, targets[st:en])

        grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)
        
        # accumulate gradients computed in previous step and call backwards
        z, count = 0,0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if grad_w[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    # compute final sensitivity metric and put in grads
    def grasp(layer):
        if layer.weight.grad is not None:
            return -layer.weight.data * layer.weight.grad   # -theta_q Hg
            #NOTE in the grasp code they take the *bottom* (1-p)% of values
            #but we take the *top* (1-p)%, therefore we remove the -ve sign
            #EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
        else:
            return torch.zeros_like(layer.weight)
    
    grads = get_layer_metric_array(net, grasp, mode)
    return grads

def compute_plain_per_weight(net, inputs, targets, loss_fn, mode = 'param', split_data=1):
    net.zero_grad()
    N = inputs.shape[0]
    # ['input_ids']
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
        # inputs = inputs[st:en]
        # {k: v[st:en].cuda() for k, v in inputs.items()}
        outputs = net.forward( inputs[st:en])
        # .logits
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def plain(layer):
        if layer.weight.grad is not None:
            return layer.weight.grad * layer.weight
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, plain, mode)
    return grads_abs

def compute_synflow_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None):

    device = inputs.device

    #convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    #convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)
    # import ipdb; ipdb.set_trace()
    # Compute gradients with input of 1s 
    net.zero_grad()
    net.double()
    # ['input_ids']
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)
    # inputs = {'input_ids':torch.ones([1] + input_dim, 	dtype = torch.long).to(device)}
    # print(inputs.shape)
    # import ipdb; ipdb.set_trace()
    output = net.forward(inputs)
    # .logits
    torch.sum(output).backward() 

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)
    return grads_abs

metric_dict = {
    'grad_norm' : get_grad_norm_arr,
    'snip':compute_snip_per_weight , 
    'grasp': compute_grasp_per_weight, 
    'fisher': compute_fisher_per_weight, 
    'plain' : compute_plain_per_weight,
    'synflow': compute_synflow_per_weight 
}

