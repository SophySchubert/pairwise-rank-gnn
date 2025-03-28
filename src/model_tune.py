import torch
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
import numpy as np
from datetime import datetime
import sys
from shutil import copyfile
import pickle
import os.path
import os
import tempfile
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import os
import tempfile
from pathlib import Path
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool, EdgeConv, GraphConv
import torch.nn as nn

from misc import setup_experiment, setup_logger, _read_config
from data.load import get_data
from data.loader import CustomDataLoader
from models.torch_gnn import RankGNN


class Net(nn.Module):
    def __init__(self, config_units=32, dropout=0.5):
        super(Net, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = GCNConv(9, config_units)
        self.conv2 = GCNConv(config_units, 32)
        self.fc1 = Linear(32, 32)
        self.fc = Linear(32, 1)  # Output 1 for regression
        self.dropout = Dropout(dropout)

    def pref_lookup(self, util, idx_a, idx_b):
        util = util.squeeze()
        idx_a = idx_a.to(torch.int64)
        idx_b = idx_b.to(torch.int64)
        pref_a = torch.gather(util, 0, idx_a)
        pref_b = torch.gather(util, 0, idx_b)
        return pref_a, pref_b

    def forward(self, data):
        # print("network")
        x, edge_index, batch, idx_a, idx_b = data.x, data.edge_index, data.batch, data.idx_a, data.idx_b
        x = x.type(torch.FloatTensor).to(self.device)
        x = self.conv1(x, edge_index)
        x = F.tanh(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.tanh(x)
        x = self.dropout(x)
        x=self.fc1(x)
        x = self.fc(x)
        x_util = global_mean_pool(x, batch)

        x_a, x_b = self.pref_lookup(x_util, idx_a, idx_b)
        out = x_b - x_a

        return out



def train_confs(config):
    myconfig = {'dataset': 'ogbg-molesol', 'mode': 'default'}
    train_dataset, valid_dataset, test_dataset, train_prefs, valid_prefs, test_prefs, test_ranking = get_data(myconfig)
    trainloader = CustomDataLoader(train_prefs,train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    valloader = CustomDataLoader(valid_prefs,valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(config['model_units'], config['model_dropout'])
    net = net.to(device)


    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data
            inputs = inputs.to(device)
            labels= inputs.y.float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs = data
                inputs = inputs.to(device)
                labels= inputs.y.float()

                outputs = net(inputs)
                total += labels.size(0)
                correct += (outputs == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": val_loss / val_steps, "accuracy": correct / total},
                checkpoint=checkpoint,
            )

    print("Finished Training")

def test_accuracy(net, device="cpu"):
    myconfig = {'dataset': 'ogbg-molesol', 'mode': 'default'}
    train_dataset, valid_dataset, test_dataset, train_prefs, valid_prefs, test_prefs, test_ranking = get_data(myconfig)
    trainloader = CustomDataLoader(train_prefs,train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    testloader = CustomDataLoader(valid_prefs,valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            total += labels.size(0)
            correct += (outputs == labels).sum().item()

    return correct / total

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    data_dir = os.path.abspath("./data")
    config = {
        "model_units": tune.choice([32,64,128,]),
        "model_dropout": tune.uniform(0.1, 0.9),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32,64,128,256,512,1024,10000,14000]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_confs),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_acc = test_accuracy(best_trained_model, device)
        print("Best trial test set accuracy: {}".format(test_acc))



if __name__ == '__main__':
    main(num_samples=100, max_num_epochs=50, gpus_per_trial=0)


