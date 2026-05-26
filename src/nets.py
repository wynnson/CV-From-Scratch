import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        """Math:
        -> in: (1, 32, 32), kernel = 5, out channels = 6, p = 0, s = 1
        -> output_size = (32 - 5 + 2(0) / 1) + 1 = 28 -> 6, 28, 28
        -> pool with 2x2 -> 6, 14, 14
        -> conv, kernel = 5, output channels = 16 -> (14 - 5) + 1 = 10 -> 16, 10, 10
        -> pool with 2x2 -> 16, 5, 5
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        
        # Pytorch way 
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),

            nn.Flatten(),
            
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),

            nn.Linear(120, 84),
            nn.Tanh(),

            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        """Does a forward pass through the NN.

        Args:
            x: input image of (1, 32, 32)

        Returns:
            logits for each class
        """
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def torch_forward(self, x):
        """Performs forward pass with torch's NN sequential.

        Args:
            x: input image of (1, 32, 32)

        Returns:
            logits for each class
        """
        return self.net(x)


class VGGBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """Does a forward pass through the VGG block.

        Args:
            x: input image

        Returns:
            feature map: (batch, channels, h, w)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x)
    
class MiniVGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stack = nn.Sequential(
            VGGBlock(3, 32),
            VGGBlock(32, 64),
            VGGBlock(64, 128)
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Does a forward pass through the NN.

        Args:
            x: input image

        Returns:
            logits for each class
        """
        return self.head(self.stack(x))


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        """Does a forward pass through the block.

        Args:
            x: input image

        Returns:
            feature map
        """
        f_x = F.relu(self.bn1(self.conv1(x)))
        f_x = self.bn2(self.conv2(f_x))
        res = f_x + self.skip(x)
        return F.relu(res)
    
class MiniResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(32, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, num_blocks=2, stride=2)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_c, out_c, num_blocks, stride):
        """Makes a layer of ResBlocks.
        
        Args:
            in_c: input channels
            out_c: output channels
            num_blocks: ResBlocks per layer
            stride: slide amount
        """
        q = [ResBlock(in_c, out_c, stride)]
        
        for _ in range(num_blocks - 1):
            q.append(ResBlock(out_c, out_c, 1))
        
        return nn.Sequential(*q)
    
    def forward(self, x):
        """Does a forward pass through the NN.

        Args:
            x: input image

        Returns:
            logits for each class
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x


# ========== TESTING ========== #

# net = LeNet5()
# x = torch.randn(1, 1, 32, 32)
# model_A_probs = F.softmax(net.forward(x), dim=1)
# model_B_probs = F.softmax(net.torch_forward(x), dim=1)
# print(model_A_probs)
# print(model_B_probs)


# net = MiniVGG()
# x = torch.randn(1, 3, 32, 32)
# print(f"output: {net(x).shape}")
# print(f"params: {sum(p.numel() for p in net.parameters()):,}")


net = MiniResNet()
x = torch.randn(1, 3, 32, 32)
print(f"output: {net(x).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")