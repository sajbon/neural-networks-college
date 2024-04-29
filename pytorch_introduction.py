from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import torch.utils
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.utils.data


class Data(torch.utils.data.Dataset):
    def __init__(self, x, y, scalling=True):
        if not torch.is_tensor(x) and not torch.is_tensor(y):
            if scalling:
                x = StandardScaler().fit_transform(x)
            self.x = torch.from_numpy(x)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(13, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.layers(x)


# Initializing and splitting data
boston = fetch_openml(name="boston", version=1)
x = boston.data
y = boston.target
y = y.to_numpy()

data = Data(x, y)
learning_data = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True)


# Initializing MLP
mdl = MLP()


# Loss function
loss = nn.MSELoss()

# Optimalization
opt = torch.optim.Adam(mdl.parameters(), lr=0.01)


# Number of epochs
n = 10
for epoch in range(0, n):
    print("Epoch number: ", epoch + 1)
    for i, interation_data in enumerate(learning_data):
        input, output = interation_data
        input = input.float()
        output = output.float()
        output = output.reshape((output.shape[0], 1))

        # Reseting gradient optimizer to zero
        opt.zero_grad()
        ob_output = mdl(input)
        loss_idk = loss(ob_output, output)
        loss_idk.backward()
        opt.step()
