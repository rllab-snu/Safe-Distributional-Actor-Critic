import torch


@torch.jit.script
def normalize(a, minimum, maximum):
    '''
    input range: [min, max]
    output range: [-1.0, 1.0]
    '''
    temp_a = 2.0/(maximum - minimum)
    temp_b = (maximum + minimum)/(minimum - maximum)
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b


@torch.jit.script
def unnormalize(a, minimum, maximum):
    '''
    input range: [-1.0, 1.0]
    output range: [min, max]
    '''
    temp_a = (maximum - minimum)/2.0
    temp_b = (maximum + minimum)/2.0
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b


@torch.jit.script
def clip(a, minimum, maximum):
    clipped = torch.where(a > maximum, maximum, a)
    clipped = torch.where(clipped < minimum, minimum, clipped)
    return clipped


def initWeights(m, init_bias=0.0):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(init_bias, 0.01)


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, shape, activation):
        super(MLP, self).__init__()
        self.activation_fn = activation
        modules = [torch.nn.Linear(input_size, shape[0]), self.activation_fn()]
        for idx in range(len(shape)-1):
            modules.append(torch.nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
        modules.append(torch.nn.Linear(shape[-1], output_size))
        self.architecture = torch.nn.Sequential(*modules)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def forward(self, input):
        return self.architecture(input)
