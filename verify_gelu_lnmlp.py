import numpy as np
import torch

fc1_input = torch.load('fc1_input.pt').detach()
fc1_weight = torch.load('fc1_weight.pt').detach()
fc1_bias = torch.load('fc1_bias.pt').detach()
fc1_gelu_output = torch.load('fc1_gelu_out.pt').detach()

#x = gelu_input.cpu().numpy()
x = fc1_input @ fc1_weight.T + fc1_bias
x = x.cpu().numpy()
numpy_gelu_output = x * 0.5 * (1.0 + np.tanh((0.7978845608028654 * (x + 0.044715 * x * x * x))))
#torch_gelu_output = torch.nn.GELU(approximate='tanh')(x)
print('numpy_gelu_output=',numpy_gelu_output)
#print('torch_gelu_output=',torch_gelu_output)
print('fc1_gelu_output=',fc1_gelu_output)

print('norm=', np.linalg.norm(numpy_gelu_output-fc1_gelu_output.cpu().numpy()))


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

fc1_output = torch.load('fc1_out.pt').detach()
fc2_weight = torch.load('fc2_weight.pt').detach()
fc2_output_grad = torch.load('fc2_output_grad.pt').detach()
fc2_gelu_dgrad = torch.load('fc2_gelu_dgrad.pt').detach()

numpy_gelu_dgrad = gelu_backward(fc1_output.cpu().numpy(), fc2_output_grad.cpu().numpy() @ fc2_weight.cpu().numpy())
#numpy_gelu_dgrad =  fc2_output_grad.cpu().numpy() @ fc2_weight.cpu().numpy()
print('numpy_gelu_dgrad=', numpy_gelu_dgrad)
print('fc2_gelu_dgrad=', fc2_gelu_dgrad)
#numpy_gelu_input_grad = gelu_backward(x, output_grad2.cpu().numpy() @ weight2.cpu().numpy())
print('norm=', np.linalg.norm(numpy_gelu_dgrad-fc2_gelu_dgrad.cpu().numpy()))
