# 用于上传作业系统
# main.py
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F

# 答应当前库版本（训练环境中的库版本应保持与比赛环境相同）
print('torch: ', torch.__version__, sep='')
print('torchvision: ', torchvision.__version__, sep='')


# 定义神经网络模型，要与训练环境（本环境）中的一致
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = torch.nn.Linear(784, 512)
#         self.fc2 = torch.nn.Linear(512, 256)
#         self.fc3 = torch.nn.Linear(256, 10)

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


# 与训练欢迎一致
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 读取训练好的模型
model = torch.load('./model.pkl')

# 生成测试结果文件
path = './data/test'
# 保存结果
answer = []
# 循环文件，识别并保存到answer中
for f in (os.path.join(path, file) for file in os.listdir(path) if file.endswith('bmp')):
    img = Image.open(f)
    f = f.split('/')[-1]
    image = transform(img).unsqueeze(0)
    y = model(image)
    ret = torch.argmax(y, dim=1)
    # 打印每一个识别结果
    print(int(f.strip('.bmp')), int(ret))
    answer.append((int(f.strip('.bmp')), int(ret)))
# 排序
answer = sorted(answer, key=lambda a: a[0])
# 写入结果文件
with open('result.csv', 'w') as f:
    for k, v in answer:
        print("%d,%s" % (k, v), file=f)