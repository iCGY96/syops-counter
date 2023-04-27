# Synaptic OPerations (SyOPs) counter for spiking neural networks
[![Pypi version](https://img.shields.io/pypi/v/syops.svg)](https://pypi.org/project/syops/)
<!-- [![Build Status](https://travis-ci.com/iCGY96/syops-counter.svg?branch=master)](https://travis-ci.com/iCGY96/syops-counter) -->

This script is designed to compute the theoretical amount of synaptic operations 
in spiking neural networks, including accumulated (AC) and multiply-accumulate (MAC) operations. 
It can also compute the number of parameters and
print per-layer computational cost of a given network.
__This tool is still under construction. 
Comments, issues, contributions, and collaborations are all welcomed!__


Supported layers:
- Conv1d/2d/3d (including grouping)
- ConvTranspose1d/2d/3d (including grouping)
- BatchNorm1d/2d/3d, GroupNorm, InstanceNorm1d/2d/3d
- Activations (ReLU, PReLU, ELU, ReLU6, LeakyReLU, GELU)
- Linear
- Upsample
- Poolings (AvgPool1d/2d/3d, MaxPool1d/2d/3d and adaptive ones)
- LF/LIF/PLIF ([spikingjelly](https://github.com/fangwei123456/spikingjelly))

Experimental support:
- RNN, LSTM, GRU (NLH layout is assumed)
- RNNCell, LSTMCell, GRUCell
- MultiheadAttention

Requirements: Pytorch >= 1.1, torchvision >= 0.3, spikingjelly<=0.0.0.0.12

## Usage

- This script doesn't take into account `torch.nn.functional.*` operations. For an instance, if one have a semantic segmentation model and use `torch.nn.functional.interpolate` to upscale features, these operations won't contribute to overall amount of flops. To avoid that one can use `torch.nn.Upsample` instead of `torch.nn.functional.interpolate`.
- `syops` launches a given model on a random tensor or a `DataLoader` and estimates amount of computations during inference. Complicated models can have several inputs, some of them could be optional. 
	- To construct non-trivial input one can use the `input_constructor` argument of the `get_model_complexity_info`. `input_constructor` is a function that takes the input spatial resolution as a tuple and returns a dict with named input arguments of the model. Next this dict would be passed to the model as a keyword arguments.
	- To construct a `DataLoader` input one can use the `dataLoader` argument of the `get_model_complexity_info` based on `torch.utils.data.DataLoader`. The number of computations would be estimated based on the input fire rate of spike signals.
- `verbose` parameter allows to get information about modules that don't contribute to the final numbers.
- `ignore_modules` option forces `syops` to ignore the listed modules. This can be useful
for research purposes. For an instance, one can drop all batch normalization from the counting process
specifying `ignore_modules=[torch.nn.BatchNorm2d]`.

## Install the latest version
From PyPI:
```bash
pip install syops
```

From this repository:
```bash
pip install --upgrade git+https://github.com/iCGY96/syops-counter
```

## Example
```python
import torch
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_resnet
from syops import get_model_complexity_info

dataloader = ...
with torch.cuda.device(0):
    net = spiking_resnet.spiking_resnet18(pretrained=True, spiking_neuron=neuron.IFNode, 
			surrogate_function=surrogate.ATan(), detach_reset=True)
    ops, params = get_model_complexity_info(net, (3, 224, 224), dataloader, as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity ACs:', acs))
    print('{:<30}  {:<8}'.format('Computational complexity MACs:', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
```

## Benchmark
Model             | Input Resolution | Params(M) | ACs(G)  | MACs(G) | Energy (mJ) | Acc@1       | Acc@5
---               |---               |---        |---      |---      |---          | ---         |---
spiking_resnet18  |224x224           | 11.69     | 0.10    | 0.14    | 0.734       | 62.32       | 84.05
sew_resnet18      |224x224           | 11.69     | 0.50    | 2.75    | 13.10       | 63.18       | 84.53
resnet18          |224x224           | 11.69     | 0.00    | 1.82    | 8.372       | 69.76       | 89.08


* ACs(G) - The theoretical amount of accumulated operations based on spike signals.
* MACs(G) - The theoretical amount of multiply-accumulate operations based on non-spike signals.
* Energy(mJ) - Energy consumption is based on 45nm technology, where AC cost 0.9pJ and MAC cost 4.6pJ.
* Acc@1 - ImageNet single-crop top-1 accuracy on validation images of the same size used during the training process.
* Acc@5 - ImageNet single-crop top-5 accuracy on validation images of the same size used during the training process.


## Acknowledgements

This repository is developed based on [ptflops](https://github.com/sovrasov/flops-counter.pytorch)