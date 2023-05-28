# 初始化工作
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import torch.nn.functional as F

print('torch: ', torch.__version__, sep='')
print('torchvision: ', torchvision.__version__, sep='')

# 自定义数据集类型
class MnistDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.images = [os.path.join(self.root, path) for path in os.listdir(self.root) if path.endswith('bmp')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path = self.images[item]  # 图像索引，获取单张图像路径
        image = Image.open(image_path)
        _, image_name = os.path.split(image_path)
        label = image_name.split('.')[-2]
        label = int(label[-1])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label
    
# 网络定义（3层全链接网络）
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(784, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = MnistDataset('./data/train', transform=transform)
test_dataset = MnistDataset('./data/test', transform=transform)

# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 初始化模型和优化器
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 训练模型
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
# 测试模型
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    

# 训练5轮
for epoch in range(1, 80):
    train(epoch)
    test()

# 保存模型到文件 model.pkl
torch.save(model, './model.pkl')