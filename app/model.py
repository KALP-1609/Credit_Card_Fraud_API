import torch.nn as nn

class FraudANN(nn.Module):
    def __init__(self,input_size,hidden1,hidden2,dropout1,dropout2,):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size,hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),

            nn.Linear(hidden1,hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),

            nn.Linear(hidden2,1),
        )

    def forward(self,x):
        return self.network(x)