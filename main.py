import os
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import monitor
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.use('Agg')


def train(epoch, loader, device, disc_iters, optim_disc, optim_gen,
          discriminator, generator, disc_monitor_manager, gen_monitor_manager, Z_dim,
          scheduler_d, scheduler_g):
    iters = 0

    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = Variable(data.to(device)), Variable(target.to(device))

        # update discriminator
        for _ in range(disc_iters):
            z = Variable(torch.randn(args.batch_size, Z_dim).to(device))
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            # Hinge loss
            disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(
                1.0 + discriminator(generator(z))).mean()

            dict_to_disc = {}
            for layer_id, layer in discriminator.spec_norm_layers.items():
                dict_to_disc[layer_id] = layer.saves_sigma

            dict_to_disc['disc_loss'] = disc_loss
            disc_monitor_manager.add_batch_dict(dict_to_disc, args.batch_size)
            disc_loss.backward()
            optim_disc.step()

        z = Variable(torch.randn(args.batch_size, Z_dim).to(device))

        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        # Hinge loss
        gen_loss = -discriminator(generator(z)).mean()
        gen_monitor_manager.add_batch_dict({'gen_loss': gen_loss}, args.batch_size)

        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            print('disc_loss: ', disc_loss.item(), 'gen_loss:', gen_loss.item())

        disc_monitor_manager.write(iters)
        gen_monitor_manager.write(iters)

        iters += 1

        disc_monitor_manager.reset()
        gen_monitor_manager.reset()

    scheduler_d.step()
    scheduler_g.step()


def evaluate(epoch, generator, fixed_z):
    samples = generator(fixed_z).cpu().data.numpy()[:64]

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1, 2, 0)) * 0.5 + 0.5)

    if not os.path.exists(f'{args.out_dir}/'):
        os.makedirs(f'{args.out_dir}/')

    plt.savefig('{}/{}.png'.format(args.out_dir, str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)


def main(args):
    device = torch.device(args.device)
    disc_iters = args.disc_iters  # discriminator updates per one generator update

    if args.dataset == 'cifar':

        import model_resnet_spec

        loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data/', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
            batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

        Z_dim = 28

    else:
        return

    discriminator = model_resnet_spec.Discriminator().to(device)
    generator = model_resnet_spec.Generator(Z_dim).to(device)
    fixed_z = Variable(torch.randn(args.batch_size, Z_dim).to(device))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0, 0.9))
    optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))

    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

    writer = SummaryWriter(log_dir='./tensor_logs/' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"))

    disc_monitor_manager = monitor.MonitorsManager(writer,
                                                   batch_var_mean=True, hist=False, hist_mode='auto',
                                                   monitor_id='disc')

    gen_monitor_manager = monitor.MonitorsManager(writer,
                                                  batch_var_mean=True, hist=False, hist_mode='auto',
                                                  monitor_id='gen')

    for epoch in range(args.epochs):
        train(epoch, loader, device, disc_iters, optim_disc, optim_gen,
              discriminator, generator, disc_monitor_manager, gen_monitor_manager,
              Z_dim, scheduler_d, scheduler_g)
        evaluate(epoch, generator, fixed_z)
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--disc_iters', type=int, default=5)

    args = parser.parse_args()

    main(args)
