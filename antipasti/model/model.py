import torch
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout

class ANTIPASTI(Module):
    def __init__(
            self,
            n_filters=16, 
            filter_size=4,
            pooling_size=2,  # Increased pooling size for more aggressive down-sampling
            input_shape=281,
            l1_lambda=0.001,  # Adjusted regularization strength
            mode='full',
            dropout_rate=0.5  # Added dropout rate parameter
    ):
        super(ANTIPASTI, self).__init__()
        self.mode = mode
        if self.mode == 'full':
            self.conv1 = Conv2d(1, n_filters, filter_size)
            self.bn1 = BatchNorm2d(n_filters)  
            self.relu = ReLU()
            self.pool = MaxPool2d((pooling_size, pooling_size))
            self.dropout = Dropout(dropout_rate)  
            self.conv2 = Conv2d(n_filters, n_filters * 2, filter_size)  
            self.bn2 = BatchNorm2d(n_filters * 2)  
            reduced_size = (input_shape - filter_size + 1) // pooling_size
            reduced_size = (reduced_size - filter_size + 1) // pooling_size  
            self.fc1 = Linear(n_filters * 2 * reduced_size ** 2, 1, bias=False)
        else:
            self.fc1 = Linear(input_shape ** 2, 1, bias=False)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        inter = x
        if self.mode == 'full':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.pool(x)
            inter = x
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return torch.sigmoid(x).float(), inter  # Using sigmoid to bound the output

    def l1_regularization_loss(self):
        l1_loss = torch.tensor(0.0)
        for param in self.parameters():
            l1_loss += torch.norm(param, p=1)
        return self.l1_lambda * l1_loss
