import os
import shutil
import random
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

torch.manual_seed(0)

class_names = ['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting']
root_dir = './Database'
#Abuse,Arrest,Arson,Assault,Burglary,Explosion,Fighting,Normal
source_dirs = ['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting']



            
class VideoDataSet(torch.utils.data.Dataset):
    def __init__(self, image_dirs,transform):
        def get_images(class_name):
            print(class_name)
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')]
            print(f'Found {len(images)}{class_name}')
            return images
        self.images={}
        self.class_names=['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting']
        for c in self.class_names:
            self.images[c]=get_images(c)
        self.image_dirs=image_dirs
        self.transform=transform
    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])
    def __getitem__(self, index):
        class_name=random.choice(self.class_names)
        index=index%len(self.images[class_name])
        image_name=self.images[class_name][index]
        image_path =os.path.join(self.image_dirs[class_name], image_name)
        image=Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)


train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])

#['NORMAL', 'Abuse ', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting']

train_dirs = {
    'Normal': 'Database/Normal',
    'Abuse': 'Database/Abuse',
    'Arrest': 'Database/Arrest',
    'Arson': 'Database/Arson',
    'Assault': 'Database/Assault',
    'Burglary': 'Database/Burglary',
    'Explosion': 'Database/Explosion',
    'Fighting': 'Database/Fighting'
    
    
}
train_dataset=VideoDataSet(train_dirs, train_transform)
test_dirs = {
    'Normal': 'Database/Test/Normal',
    'Abuse': 'Database/Test/Abuse',
    'Arrest': 'Database/Test/Arrest',
    'Arson': 'Database/Test/Arson',
    'Assault': 'Database/Test/Assault',
    'Burglary': 'Database/Test/Burglary',
    'Explosion': 'Database/Test/Explosion',
    'Fighting': 'Database/Test/Fighting'
}
test_dataset = VideoDataSet(test_dirs, test_transform)

batch_size=6
dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print('Num of training batches', len(dl_train))
print('Num of test batches', len(dl_test))


class_names=train_dataset.class_names
def show_images(images, labels, preds):
    plt.figure(figsize=(8,4))
    for i, image in enumerate(images):
        plt.subplot(1,6,i+1, xticks=[], yticks=[])
        image=image.numpy().transpose((1,2,0))
        mean=np.array([0.485,0.456,0.406])
        std= np.array([0.229, 0.224, 0.225])
        image=image*std/mean
        image=np.clip(image,0.,1.)
        plt.imshow(image)
        col = 'green' if preds[i]==labels[i] else 'red'
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()
##########################
def show_images1(images, labels, preds):
    plt.figure(figsize=(8,4))
    for i, image in enumerate(images):
        plt.subplot(1,6,i+1, xticks=[], yticks=[])
        image=image.numpy().transpose((1,2,0))
        mean=np.array([0.485,0.456,0.406])
        std= np.array([0.229, 0.224, 0.225])
        image=image*std/mean
        image=np.clip(image,0.,1.)
        plt.imshow(image)
        col = 'green' if preds[i]==labels[i] else 'red'
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.xlabel(f'{class_names[int(preds[i].numpy())]}', color=col)
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()

images, labels =next(iter(dl_train))
show_images(images, labels, labels)


images, labels =next(iter(dl_test))
show_images(images, labels, labels)

propNN =torchvision.models.resnet18(pretrained=True)
print(propNN)

propNN.fc=torch.nn.Linear(in_features=512, out_features=8)
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(propNN.parameters(), lr=3e-5)
def show_preds():
    propNN.eval()
    images, labels =next(iter(dl_test))
    outputs = propNN(images)
    _, preds=torch.max(outputs, 1)
    show_images1(images, labels, preds)
#show_preds()

def train(epochs):
    print('Starting training..')
    for e in range(0, epochs):
        print(f'Starting epoch {e+1}/{epochs}')
        print('='*20)
        train_loss=0
        propNN.train()
        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs=propNN(images)
            loss=loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss=loss.item()
            if train_step%20==0:
                print('Evaluating at step', train_step)
                acc=0.
                val_loss=0.
                propNN.eval()
                for val_step,(images, labels) in enumerate(dl_test):
                    outputs=propNN(images)
                    loss=loss_fn(outputs, labels)
                    val_loss+=loss.item()
                    _,preds=torch.max(outputs, 1)
                    acc+=sum(preds==labels).numpy()
                val_loss/=(val_step+1)
                acc=acc/len(test_dataset)
                print(f'Val loss: {val_loss:.4f}, Acc: {acc:.4f}')
                #show_preds()
                propNN.train()
                print('Accuracy:::',acc)
                
        train_loss/=(train_step+1)
        print(f'Training loss: {train_loss:.4f}')
train(epochs=1)
print('Result....')
show_preds()
