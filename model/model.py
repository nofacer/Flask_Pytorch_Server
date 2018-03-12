
import torch.nn as nn


class ShapeDetectNetwork(nn.Module):
    def __init__(self):
        super(ShapeDetectNetwork,self).__init__()
        self.conv = nn.Sequential(
            #3@32*32
            nn.Conv2d(3, 8, 5),
            #8@28*28
            nn.ReLU(),
            nn.MaxPool2d(2),
            #8@14*14
            nn.Conv2d(8, 32, 3),
            #32@12*12
            nn.ReLU(),
            nn.MaxPool2d(2),
            #32@6*6
            nn.Conv2d(32, 64, 3),
            #64@4*4
            nn.ReLU(),
            )

        self.fc=nn.Sequential(
            nn.Linear(64*4*4,1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
            nn.Sigmoid(),
            )



    def forward(self,x):

        o1=self.conv(x)
        o1=o1.view(o1.size()[0],-1)
        o2=self.fc(o1)

        return o2


