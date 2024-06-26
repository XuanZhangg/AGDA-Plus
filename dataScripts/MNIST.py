
import torch
import torchvision
from torchvision import transforms
import pickle

train_set = torchvision.datasets.MNIST('/.data', train=True, download=True,
                             transform = transforms.Compose([
                             transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
                             )

#Limit the data to digits 0,2,3, convert data type to float32, make shape of [28,28] to [1,28,28]
idx = (train_set.targets == 0) | (train_set.targets == 2) | (train_set.targets == 3) 
train_set.targets = train_set.targets[idx]
train_set.data = train_set.data[idx].type(torch.float32).unsqueeze(1)

#change the targets to use onehot encoding
idx2 = train_set.targets == 2
idx3 = train_set.targets == 3
train_set.targets[idx2] = 1
train_set.targets[idx3] = 2

print('Examples of train set:') 
print(train_set.targets[0:10])
print(train_set.data.shape)

file_name = './data/MNIST/MNIST'
with open(file_name , "wb") as fp:  
    pickle.dump(train_set, fp)

