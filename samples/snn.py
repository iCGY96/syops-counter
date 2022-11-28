import argparse
import sys

import torch
import torchvision
from torchvision import transforms

from spikingjelly.clock_driven import surrogate, neuron, functional
import spiking_resnet
import sew_resnet
from torchvision import models as models

from syops import get_model_complexity_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='syops sample script')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to store the model.')
    parser.add_argument('--result', type=str, default=None)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        args.root,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=20, drop_last=True, 
            num_workers=4, pin_memory=True)

    net = spiking_resnet.spiking_resnet18(T=4)
    # net = sew_resnet.sew_resnet152(T=4, connect_f='ADD')

    model_without_ddp = net
    checkpoint = torch.load(args.resume, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model_without_ddp.load_state_dict(new_state_dict)
    
    # net = models.resnet18()

    if torch.cuda.is_available():
        net.cuda(device=args.device)

    ops, params = get_model_complexity_info(net, (3, 224, 224), dataloader,
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             ignore_modules=[torch.nn.BatchNorm2d],
                                             ost=ost)
    print('{:<30}  {:<8}'.format('Computational complexity OPs:', ops[0]))
    print('{:<30}  {:<8}'.format('Computational complexity ACs:', ops[1]))
    print('{:<30}  {:<8}'.format('Computational complexity MACs:', ops[2]))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
