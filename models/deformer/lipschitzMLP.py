import torch
import numpy as np

def leaky_relu_init(m, negative_slope=0.2):

    gain = np.sqrt(2.0 / (1.0 + negative_slope ** 2))

    if isinstance(m, torch.nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // 2
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return

  
    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, torch.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    

    # m.weights_initialized=True

def apply_weight_init_fn(m, fn, negative_slope=1.0):

    should_initialize_weight=True
    if not hasattr(m, "weights_initialized"): #if we don't have this then we need to intiialzie
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    elif m.weights_initialized==False: #if we have it but it's set to false
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    else:
        print("skipping weight init on ", m)
        should_initialize_weight=False

    if should_initialize_weight:
        # fn(m, is_linear, scale)
        fn(m,negative_slope)
        # m.weights_initialized=True
        for module in m.children():
            apply_weight_init_fn(module, fn, negative_slope)
            
class LipshitzMLP(torch.nn.Module):

    def __init__(self, in_channels, nr_out_channels_per_layer, last_layer_linear):
        super(LipshitzMLP, self).__init__()


        self.last_layer_linear=last_layer_linear
     

        self.layers=torch.nn.ModuleList()
        # self.layers=[]
        for i in range(len(nr_out_channels_per_layer)):
            cur_out_channels=nr_out_channels_per_layer[i]
            self.layers.append(  torch.nn.Linear(in_channels, cur_out_channels)   )
            in_channels=cur_out_channels
        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        if last_layer_linear:
            leaky_relu_init(self.layers[-1], negative_slope=1.0)

        #we make each weight separately because we want to add the normalize to it
        self.weights_per_layer=torch.nn.ParameterList()
        self.biases_per_layer=torch.nn.ParameterList()
        for i in range(len(self.layers)):
            self.weights_per_layer.append( self.layers[i].weight  )
            self.biases_per_layer.append( self.layers[i].bias  )

        self.lipshitz_bound_per_layer=torch.nn.ParameterList()
        for i in range(len(self.layers)):
            max_w= torch.max(torch.sum(torch.abs(self.weights_per_layer[i]), dim=1))
            #we actually make the initial value quite large because we don't want at the beggining to hinder the rgb model in any way. A large c means that the scale will be 1
            c = torch.nn.Parameter(  torch.ones((1))*max_w*2 ) 
            self.lipshitz_bound_per_layer.append(c)






        self.weights_initialized=True #so that apply_weight_init_fn doesnt initialize anything

    def normalization(self, w, softplus_ci):
        absrowsum = torch.sum(torch.abs(w), dim=1)
        # scale = torch.minimum(torch.tensor(1.0), softplus_ci/absrowsum)
        # this is faster than the previous line since we don't constantly recreate a torch.tensor(1.0)
        scale = softplus_ci/absrowsum
        scale = torch.clamp(scale, max=1.0)
        return w * scale[:,None]

    def lipshitz_bound_full(self):
        lipshitz_full=1
        for i in range(len(self.layers)):
            lipshitz_full=lipshitz_full*torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])

        return lipshitz_full

    def forward(self, x):

        # x=self.mlp(x)

        for i in range(len(self.layers)):
            weight=self.weights_per_layer[i]
            bias=self.biases_per_layer[i]

            weight=self.normalization(weight, torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])  )

            x=torch.nn.functional.linear(x, weight, bias)

            is_last_layer=i==(len(self.layers)-1)

            if is_last_layer and self.last_layer_linear:
                pass
            else:
                x=torch.nn.functional.gelu(x)


        return x