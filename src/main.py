import os
from pprint import pprint
import time

from tqdm import tqdm
import click
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


@click.command()
@click.option('--data-dir', type=str, default='~/Downloads/data', show_default=True)
@click.option('--distributed', is_flag=True, default=False)
@click.option('--use-cpu', is_flag=True, default=False)
@click.option('--batch-size', type=int, default=32, show_default=True)
@click.option('--num-workers', type=int, default=2, show_default=True)
@click.option('--learning-rate', type=float, default=0.1, show_default=True)
@click.option('--weight-decay', type=float, default=0.001, show_default=True)
def main(
        data_dir: str,
        distributed: bool,
        use_cpu: bool,
        batch_size: int,
        num_workers: int,
        learning_rate: float,
        weight_decay: float,
):
    if distributed:
        assert torch.cuda.is_available()
        assert not use_cpu
        assert int(os.environ['WORLD_SIZE']) > 1
        dist.init_process_group(
            backend='nccl' if dist.is_nccl_available() else 'gloo',
        )
        print(f'Process group initialized - WORLD_SIZE: {dist.get_world_size()}, RANK: {dist.get_rank()}')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = CIFAR10(data_dir, train=False, download=True, transform=transform)

    net = resnet18(num_classes=len(dataset.classes))
    if distributed:
        global_rank = dist.get_rank()
        local_rank = torch.cuda.device_count() % global_rank
        net = DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
        device = local_rank
    else:
        device = torch.cuda.current_device() if torch.cuda.is_available() and not use_cpu else torch.device('cpu')
        net = net.to(device)

    print(f'Device - {device}')

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    t0 = time.perf_counter()
    print('Training starting...')

    for imgs, lbls in tqdm(dataloader):
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        optimizer.zero_grad()

        preds = net(imgs)

        losses = criterion(preds, lbls)

        losses.backward()
        optimizer.step()

    t1 = time.perf_counter()
    print('Training complete')

    print(f'Elapsed time: {t1 - t0}')


if __name__ == '__main__':
    main()
