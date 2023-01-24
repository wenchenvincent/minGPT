import sys
import argparse
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from transformer_engine import pytorch as te

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed

set_seed(3407)

import pickle

class SortDataset(Dataset):
    """ 
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits
    
    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.num_digits
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1

    def __getitem__(self, idx):
        
        # use rejection sampling to generate an input example from the desired split
        while True:
            # generate some random integers
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            # half of the time let's try to boost the number of examples that 
            # have a large number of repeats, as this is what the model seems to struggle
            # with later in training, and they are kind of rate
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    # too many unqiue digits, re-sample
                    continue
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break # ok
        
        # solve the task: i.e. sort
        sol = torch.sort(inp)[0]

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:self.length-1] = -1
        return x, y

def batch_end_callback(trainer):
    '''
    print('transform.mlp.c_fc input activation =', input_activations['transformer.mlp.c_fc'])
    print('transform.mlp.c_fc weight =', weights['transformer.mlp.c_fc'])
    print('transform.mlp.c_fc bias =', biases['transformer.mlp.c_fc'])
    print('transform.mlp.c_fc output activation =', output_activations['transformer.mlp.c_fc'])

    print('transform.mlp.c_fc output gradient =', output_gradients['transformer.mlp.c_fc'])
    print('transform.mlp.c_fc input gradient =', input_gradients['transformer.mlp.c_fc'])
    print('transform.mlp.c_fc weight gradient =', weight_gradients['transformer.mlp.c_fc'])
    print('transform.mlp.c_fc bias gradient =', bias_gradients['transformer.mlp.c_fc'])
    '''
    #print('transform.mlp.c_fc input activation =', input_activations['transformer.mlp.c_fc'])
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        '''
        torch.save(trainer.input_activations['transformer.mlp.c_fc'], 'input.pt')
        torch.save(trainer.weights['transformer.mlp.c_fc'], 'weight.pt')
        torch.save(trainer.biases['transformer.mlp.c_fc'], 'bias.pt')
        torch.save(trainer.output_activations['transformer.mlp.c_fc'], 'output.pt')

        torch.save(trainer.output_gradients['transformer.mlp.c_fc'], 'output_grad.pt')
        torch.save(trainer.input_gradients['transformer.mlp.c_fc'], 'input_grad.pt')
        torch.save(trainer.weight_gradients['transformer.mlp.c_fc'], 'weight_grad.pt')
        torch.save(trainer.bias_gradients['transformer.mlp.c_fc'], 'bias_grad.pt')
        sys.exit(0)
        '''

def eval_split(model, trainer, trainset, testset, split, max_batches):
    dataset = {'train':trainset, 'test':testset}[split]
    n = trainset.length # naugy direct access shrug
    results = []
    mistakes_printed_already = 0
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y = y.to(trainer.device)
        # isolate the input pattern alone
        inp = x[:, :n]
        sol = y[:, -n:]
        # let the model sample the rest of the sequence
        cat = model.generate(inp, n, do_sample=False) # using greedy argmax, not sampling
        sol_candidate = cat[:, n:] # isolate the filled in sequence
        # compare the predicted sequence to the true sequence
        correct = (sol == sol_candidate).all(1).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if not correct[i] and mistakes_printed_already < 3: # only print up to 5 mistakes to get a sense
                mistakes_printed_already += 1
                print("GPT claims that %s sorted is %s but gt is %s" % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist()))
        if max_batches is not None and b+1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
    return rt.sum()

############################################################################
###  Auxiliary functions for debugging
############################################################################

input_activations = {}
output_activations = {}
weights = {}
biases = {}
## Hook to record activation tensors in the forward pass
def get_activation(name):
    def forward_hook(module, input, output):
        input_activations[name] = input[0].detach()
        output_activations[name] = output.detach()
        weights[name] = module.weight.detach()
        biases[name] = module.bias.detach()
    return forward_hook

output_gradients = {}
input_gradients = {}
weight_gradients = {}
bias_gradients = {}
## Hook to record gradient tensors in the backward pass
def get_gradient(name):
    def backward_hook(module, grad_input, grad_output):
        ## grad_output and grad_input are tuples
        ## grad_input are gradients wrt inputs of the layer
        ## grad_output are gradients wrt outputs of the layer
        output_gradients[name] = grad_output[0].detach()
        input_gradients[name] = grad_input[0].detach()
        weight_gradients[name] = module.weight.grad.detach()
        bias_gradients[name] = module.bias.grad.detach()
    return backward_hook

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch minGPT Example")
    parser.add_argument(
        "--miters",
        type=int,
        default=3000,
        metavar="N",
        help="number of epochs to train (default: 2000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--use-amp", action="store_true", default=False, help="Use AMP training"
    )
    parser.add_argument(
        "--use-fp8", action="store_true", default=False, help="Use FP8 training"
    )
    parser.add_argument(
        "--use-te", action="store_true", default=False, help="Use Transformer Engine"
    )
    parser.add_argument(
        "--ln-mlp", action="store_true", default=False, help="Use Transformer Engine LayerNormMLP"
    )
    args = parser.parse_args()
    # use_cuda = torch.cuda.is_available()
    # print an example instance of the dataset
    train_dataset = SortDataset('train')
    test_dataset = SortDataset('test')
    x, y = train_dataset[0]
    for a, b in zip(x,y):
        print(int(a),int(b))


    # create a GPT instance
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()

    if args.use_te:
        model_config.use_te = True
        if args.ln_mlp:
            model_config.ln_mlp = True
    if args.use_amp:
        model_config.use_amp = True
    if args.use_fp8:
        model_config.use_fp8 = True

    model = GPT(model_config)

    '''
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    '''

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.learning_rate = args.lr
    # the model we're using is so small that we can go a bit faster
    train_config.max_iters = args.miters
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, train_dataset)
    

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    # now let's perform some evaluation
    model.eval()

    # run a lot of examples from both train and test through the model and verify the output correctness
    with torch.no_grad():
        train_score = eval_split(model, trainer, train_dataset, test_dataset, 'train', max_batches=50)
        test_score  = eval_split(model, trainer, train_dataset, test_dataset, 'test',  max_batches=50)

    # let's run a random given sequence through the model as well
    n = train_dataset.length # naugy direct access shrug
    inp = torch.tensor([[0, 0, 2, 1, 0, 1]], dtype=torch.long).to(trainer.device)
    assert inp[0].nelement() == n
    with torch.no_grad():
        cat = model.generate(inp, n, do_sample=False)
    sol = torch.sort(inp[0])[0]
    sol_candidate = cat[:, n:]
    print('input sequence  :', inp.tolist())
    print('predicted sorted:', sol_candidate.tolist())
    print('gt sort         :', sol.tolist())
    print('matches         :', bool((sol == sol_candidate).all()))


if __name__ == "__main__":
    main()
