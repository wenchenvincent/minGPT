import numpy as np
import torch

input_ = torch.load('input.pt')
weight = torch.load('weight.pt')
bias = torch.load('bias.pt')
output = torch.load('output.pt')

print('input shape=', input_.shape)
print('weight shape=', weight.shape)
print('bias shape=', bias.shape)
print('output shape=', output.shape)
#np_output = input_ @ weight.T + bias

#print('np_output=', np_output)
torch_output = torch.nn.functional.linear(input_, weight, bias)

print('output=', output)
print('torch_output=', torch_output)

print('norm=', np.linalg.norm((torch_output-output).cpu().numpy()))


output_grad = torch.load('output_grad.pt')
input_grad = torch.load('input_grad.pt')
bias_grad = torch.load('bias_grad.pt')
weight_grad = torch.load('weight_grad.pt')

print('output_grad shape=', output_grad.shape)
print('input_ shape=', input_.shape)

output_grad_reshaped = output_grad.reshape((-1, output_grad.shape[-1]))
input_reshaped = input_.reshape((-1, input_.shape[-1]))
weight_grad_reshaped = weight_grad.reshape((-1, weight_grad.shape[-1]))

np_weight_grad_reshaped = output_grad_reshaped.cpu().numpy().T @ input_reshaped.cpu().numpy()
print('np_weight_grad=', np_weight_grad_reshaped )
print('weight_grad=', weight_grad_reshaped)
print(np.linalg.norm(np_weight_grad_reshaped - weight_grad_reshaped.cpu().numpy()))

np_input_grad = output_grad.cpu().numpy() @ weight.cpu().numpy()
print('np_input_grad=', np_input_grad)
print('input_grad=', input_grad)
print(np.linalg.norm(np_input_grad-input_grad.cpu().numpy()))

np_bias_grad = output_grad.cpu().numpy().sum(axis=(0,1))
print('np_bias_grad=', np_bias_grad)
print('bias_grad=', bias_grad)
print(np.linalg.norm(np_bias_grad-bias_grad.cpu().numpy()))
