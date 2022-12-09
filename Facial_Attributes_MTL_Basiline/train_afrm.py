import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataset.CelebA import CelebA
from dataset.config import *
# from model.resnet import resnet18, resnet50
import os
from torch.autograd import Variable
import argparse
from torch.optim.lr_scheduler import *
import torchvision
from model.resnet_modulated import *
from model.alexnet_modulated import *
from model.lenet_modulated import *
from model.convnet_modulated import *

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=32)
parser.add_argument('--nepoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr-scheduler', type=str, default=None)
# parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--gpu', type=str, default='7', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--filename', type=str, help='test_model')
parser.add_argument('--model', type=str, default='resnet18', help='test_model')

opt = parser.parse_args()
print(opt)

model_name = opt.model
assert model_name in ["resnet18", "alexnet", "lenet", "convnet"]
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

data_root = "/data/alexsun/CelebA"
trainset = CelebA(f'{data_root}/list_eval_partition.txt', f'{data_root}/list_attr_celeba.txt', '0',
                  f'{data_root}/img_align_celeba/', transform_train)
# trainset = torchvision.datasets.CelebA(data_root, split="train", transform=transform_train, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)

# valset = CelebA('./data/list_eval_partition.txt', './data/list_attr_celeba.txt', '1',
#                   './data/img_align_celeba/', transform_val)
valset = CelebA(f'{data_root}/list_eval_partition.txt', f'{data_root}/list_attr_celeba.txt', '1',
                  f'{data_root}/img_align_celeba/', transform_val)
valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)

print(opt.pretrained)
if model_name == "alexnet":
    model = AlexNet(num_classes = NUM_CLASS)
elif model_name == "lenet":
    model = LeNet5(num_classes = NUM_CLASS)
elif model_name == "convnet":
    model=ConvNet(num_classes = NUM_CLASS)
elif model_name == "resnet18":
    model=resnet18(pretrained=opt.pretrained)
    model.fc=nn.Linear(512,NUM_CLASS)
#model = resnet50(pretrained=True, num_classes=40)
# model=resnet50(pretrained=True)
model.cuda()
criterion = nn.MSELoss(reduce=True)
optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
if opt.lr_scheduler == "cosine":
    scheduler = CosineAnnealingLR(optimizer, T_max = opt.nepoch)
else:
    scheduler = StepLR(optimizer, step_size=3)

def sanity_check(model):
    vals = [float(x.weight[0].item()) for x in model.bn1.bn_modules]
    print(vals)
    # vals = [float(x.weight.grad[0].item()) for x in model.bn1.bn_modules]
    # print(vals)

def train(epoch):
    print('\nTrain epoch: %d' % epoch)
    scheduler.step()
    model.train()
    for batch_idx, (images, attrs) in enumerate(trainloader):
        images = Variable(images.cuda())
        attrs = Variable(attrs.cuda()).type(torch.cuda.FloatTensor)
        optimizer.zero_grad()
        # output = model(images)
        # Innerloop the task
        total_output = []
        for task_id in range(NUM_CLASS):
            output = model(images, task_id)
            output = output.unsqueeze(-1)
            total_output.append(output)
        output = torch.cat(total_output, axis = 1)
        # print(output.shape)
        loss = criterion(output, attrs)
        loss.backward()
        optimizer.step()
        # sanity_check(model)
        # print(model.bn1.bn_modules)
        if batch_idx%100==0:
            print('[%d/%d][%d/%d] loss: %.4f' % (epoch, opt.nepoch, batch_idx, len(trainloader), loss.mean()))



def test(epoch):
    print('\nTest epoch: %d' % epoch)
    model.eval()
    # correct = torch.FloatTensor(40).fill_(0)
    correct = torch.FloatTensor(NUM_CLASS).fill_(0)
    total = 0
    with torch.no_grad():
        for batch_idx, (images, attrs) in enumerate(valloader):
            images = Variable(images.cuda())
            attrs = Variable(attrs.cuda()).type(torch.cuda.FloatTensor)
            # output = model(images)
            total_output = []
            for task_id in range(NUM_CLASS):
                output = model(images, task_id)
                output = output.unsqueeze(-1)
                total_output.append(output)
            output = torch.cat(total_output, axis = 1)
            com1 = output > 0
            com2 = attrs > 0
            correct.add_((com1.eq(com2)).data.cpu().sum(0).type(torch.FloatTensor))
            total += attrs.size(0)
    print(correct / total)
    print(torch.mean(correct / total))


print(f"Final Saved Model Path: /data/alexsun/ckp/{opt.filename}.pth")
for epoch in range(0, opt.nepoch):
    train(epoch)
    test(epoch)
# torch.save(model.state_dict(), 'ckp/model_naive.pth')
torch.save(model.state_dict(), f'/data/alexsun/ckp/{opt.filename}.pth')
