import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class BasicBlockNoSkip(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10, n=9):
        super().__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.stage1 = self._make_layers(16, n, stride=1)
        self.stage2 = self._make_layers(32, n, stride=2)
        self.stage3 = self._make_layers(64, n, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layers(self, planes, num_blocks, stride):
        layers = [BasicBlockNoSkip(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlockNoSkip(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class ResNet(nn.Module):
    def __init__(self,num_classes=10,n=9):
        super().__init__()
        self.n=n

        #First conv layer
        self.conv1=nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        
        #3 Stages of Residual Blocks
        self.stage1=self._make_layers(16,16,num_blocks=self.n,stride=1)
        self.stage2=self._make_layers(16,32,num_blocks=self.n,stride=2)
        self.stage3=self._make_layers(32,64,num_blocks=self.n,stride=2)

        #Remaining layers for Classification
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(64,num_classes)

        # Weight initialization 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self,in_channel,out_channel,num_blocks,stride):
        layers=[]
        #First block(for downsampling)
        layers.append(BasicBlockNoSkip(in_channel,out_channel,stride)) # Change to BasicBlock to use standard resnet with skip
        #Remaining
        for _ in range(1,num_blocks):
            layers.append(BasicBlockNoSkip(out_channel,out_channel,stride=1)) # Change to BasicBlock to use standard resnet with skip
        return nn.Sequential(*layers)

    def forward(self,x):
        out=self.relu(self.bn(self.conv1(x)))
        out=self.stage1(out)
        out=self.stage2(out)
        out=self.stage3(out)
        out=self.pool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out
        