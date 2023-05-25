import argparse

import torch
import torch.optim as optim

from dataset import Dataset
from model import Model
from utils import calc_metrics


def train_model(e, indexes, labels):
    print('-------- Epoch ' + str(e + 1) + ' --------')
    model.train()
    optimizer.zero_grad()
    _, loss = model(indexes, labels)
    loss.backward()
    optimizer.step()
    print('Loss: ', loss.item())


@torch.no_grad()
def test_model(indexes, labels):
    output, _ = model(indexes, labels)
    y_true_test = labels.to('cpu').numpy()
    y_score_test = output.flatten().tolist()
    calc_metrics(y_true_test, y_score_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--nl', dest='nl', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--es', dest='es', default=256, type=int, help='Embedding size')
    parser.add_argument('--nh', dest='nh', default=2, type=int, help='Number of attention heads')
    parser.add_argument('--seed', dest='seed', default=1234, type=int, help='Random seed')
    parser.add_argument('--epoch', dest='epoch', default=500, type=int, help='Training epochs')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float)
    parser.add_argument('--wd', dest='wd', default=5e-4, type=float)
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Dataset()
    train_data = dataset.fetch_train_data()
    train_indexes = [torch.tensor(train_data[0][:, 0], dtype=torch.long).to(device),
                     torch.tensor(train_data[0][:, 1], dtype=torch.long).to(device)]
    train_labels = torch.tensor(train_data[1], dtype=torch.long).to(device)
    test_data = dataset.fetch_test_data()
    test_indexes = [torch.tensor(test_data[0][:, 0], dtype=torch.long).to(device),
                    torch.tensor(test_data[0][:, 1], dtype=torch.long).to(device)]
    test_labels = torch.tensor(test_data[1], dtype=torch.long).to(device)

    model = Model(dataset, args.nl, args.es, args.nh).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.epoch):
        train_model(epoch, train_indexes, train_labels)

    print('Test results:')
    test_model(test_indexes, test_labels)
