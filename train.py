"""Training procedure for NICE.
"""

import argparse
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import nice


def split_inputs(inputs):
    features, labels = inputs
    features = features.view(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3])
    return features


def dequantize_and_scale(features):
    noise = torch.distributions.Uniform(0., 1.).sample(features.size())
    features = (features * 255. + noise) / 256.
    return features


def train(flow, trainloader, optimizer, device):
    flow.train()
    loss, i_batch = 0, 0
    for inputs in trainloader:
        i_batch += 1
        features = split_inputs(inputs)
        features = dequantize_and_scale(features)
        features = features.to(device)
        optimizer.zero_grad()
        batch_loss = -flow(features).mean()
        loss += batch_loss
        batch_loss.backward()
        optimizer.step()
    return loss / i_batch


def test(flow, testloader, filename, epoch, sample_shape, device):
    flow.eval()  # set to inference mode
    with torch.no_grad():
        samples = flow.sample(100).to(device)
        a, b = samples.min(), samples.max()
        samples = (samples-a)/(b-a+1e-10) 
        samples = samples.view(-1,sample_shape[0],sample_shape[1],sample_shape[2])
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)
        loss, i_batch = 0, 0
        for inputs in testloader:
            i_batch += 1
            features = split_inputs(inputs)
            batch_loss = -flow(features).mean()
            loss += float(batch_loss)
    return loss / i_batch


def main(args):
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # for mac
    device = "cpu"

    transform = transforms.Compose([transforms.ToTensor()])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Not a valid dataset')

    model_save_filename = '%s_' % args.dataset \
                          + 'batch%d_' % args.batch_size \
                          + 'coupling%d_' % args.coupling \
                          + 'coupling_type%s_' % args.coupling_type \
                          + 'mid%d_' % args.mid_dim \
                          + 'hidden%d_' % args.hidden \
                          + '.pt'

    flow = nice.NICE(
        prior=args.prior,
        coupling=args.coupling,
        coupling_type=args.coupling_type,
        in_out_dim=784,
        mid_dim=args.mid_dim,
        hidden=args.hidden,
        device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    train_losses = []
    test_losses = []
    filename = f"{args.dataset}_ + {args.coupling_type}_"
    shape = [1, 28, 28]

    for epoch in tqdm(range(args.epochs)):
        train_loss = train(flow, trainloader, optimizer, device)
        train_losses.append(train_loss)
        test_loss = test(flow, testloader, filename, epoch+1, shape, device)
        test_losses.append(test_loss)
        print(f"Epoch {epoch + 1} finished:  train loss: {train_loss}, test loss: {test_loss} ")
        if epoch % 5 == 0:
            torch.save(flow.state_dict(), "./models/" + model_save_filename)

    with torch.no_grad():
        fig, ax = plt.subplots()
        ax.plot(train_losses)
        ax.plot(test_losses)
        ax.set_title("Train and Test Log Likelihood Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["train loss", "test loss"])
        plt.savefig(fname="./loss/" + f"{args.dataset}_" + f"{args.coupling_type}_loss.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='fashion-mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='affine')
    parser.add_argument('--coupling',
                        help='.',
                        # type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
