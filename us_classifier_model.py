import pytorch_lightning as pl
import torch.nn as nn
import torch
from templates import *
import pickle
import numpy as np

class USSimPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(USSimPredictor, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x


def train(model, criterion, optimizer, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), f'/home/demir/Desktop/diffae_checkpoints/classifiers/us_sim_predictor_MSE_{epoch}.pth')
        for batch in train_loader:
            inputs = batch['data']
            targets = batch['label']
            targets = targets.unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

class CondDataset(Dataset):
    def __init__(self, sim_pickle_path, real_pickle_path):
        with open(sim_pickle_path, 'rb') as f:
            sim_conds_np = pickle.load(f)
            self.sim_conds = [torch.from_numpy(cond) for cond in sim_conds_np]
        with open(real_pickle_path, 'rb') as f:
            real_conds_np = pickle.load(f)
            self.real_conds = [torch.from_numpy(cond) for cond in real_conds_np]

        self.all_conds = torch.cat(self.sim_conds + self.real_conds, dim=0)
        self.conds_mean = self.all_conds.mean(dim=0)
        self.conds_std = self.all_conds.std(dim=0)

    def normalize(self, cond):
        cond = (cond - self.conds_mean) / self.conds_std
        return cond
    
    def denormalize(self, cond):
        cond = cond * self.conds_std + self.conds_mean
        return cond

    def __len__(self):
        return len(self.sim_conds) + len(self.real_conds)

    def __getitem__(self, idx):
        if idx < len(self.sim_conds):
            data = self.normalize(self.sim_conds[idx])
            label = 1 
        else:
            data = self.normalize(self.real_conds[idx - len(self.sim_conds)])
            label = 0
        return {'data': data, 'label': label}


if __name__ == "__main__":
    model = USSimPredictor(512, 1)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    dataset = CondDataset('/home/demir/Desktop/diffae-datasets/real_cond.pkl', '/home/demir/Desktop/diffae-datasets/simulated_cond.pkl')
    print(dataset.conds_mean)
    print(dataset.conds_std)
    
    
    # # print(dataset[0]['data'].shape)
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    # # for batch in train_loader:
    # #     print(batch['data'].shape)
    # train(model, criterion, optimizer, train_loader, 50)
    # # save model
    # torch.save(model.state_dict(), '/home/demir/Desktop/diffae_checkpoints/classifiers/us_sim_predictor_MSE_last.pth')
    # print("Training complete")