import numpy as np
import torch

input_ = torch.load('input.pt')
weight = torch.load('weight.pt')
bias = torch.load('bias.pt')
output = torch.load('output.pt')
weight2 = torch.load('weight2.pt')

gelu_input = torch.load('gelu_input.pt')
gelu_output = torch.load('gelu_output.pt')
gelu_input_grad = torch.load('gelu_input_grad.pt')
gelu_output_grad = torch.load('gelu_output_grad.pt')
output_grad2 = torch.load('output_grad2.pt')

print('linear output=', output)
print('gelu_input=', gelu_input)

print('input shape=', input_.shape)
print('weight shape=', weight.shape)
print('bias shape=', bias.shape)
print('output shape=', output.shape)
#np_output = input_ @ weight.T + bias


#x = gelu_input.cpu().numpy()
x = input_ @ weight.T + bias
x = x.cpu().numpy()
numpy_gelu_output = x * 0.5 * (1.0 + np.tanh((0.7978845608028654 * (x + 0.044715 * x * x * x))))
torch_gelu_output = torch.nn.GELU(approximate='tanh')(gelu_input)
print('numpy_gelu_output=',numpy_gelu_output)
print('torch_gelu_output=',torch_gelu_output)
print('gelu_output=',gelu_output)

print('norm=', np.linalg.norm(numpy_gelu_output-gelu_output.cpu().numpy()))


def gelu_backward(x, dy):
    kBeta = 0.7978845608028654
    kKappa = 0.044715
    x_sq = x * x
    x_cube = x_sq * x
    tanh_inner = np.tanh((kBeta * (x + kKappa * x_cube)))

    left = 0.5 * x
    right = 1.0 + tanh_inner

    left_derivative = 0.5 * right

    tanh_derivative = 1 - tanh_inner * tanh_inner
    inner_derivative = kBeta * (1.0 + 3.0 * kKappa * x_sq)
    right_derivative = left * tanh_derivative * inner_derivative

    return dy * (left_derivative + right_derivative)

numpy_gelu_input_grad = gelu_backward(x, output_grad2.cpu().numpy() @ weight2.cpu().numpy())
print('norm=', np.linalg.norm(numpy_gelu_input_grad-gelu_input_grad.cpu().numpy()))
