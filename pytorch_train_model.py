#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:48:35 2019

@author: pratik
"""
import numpy as np
from matplotlib import pyplot as plt
from pytorch_cnn_model import MyModel
from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

## Required together ##
#import torch
#from torchvision import datasets, transforms
#from torch.utils.data.sampler import SubsetRandomSampler
#from custom_dataloader.data_loader import get_train_valid_loader,get_test_loader
#########################

#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)

def data_normalise(data_test):
    '''
    Funcation to normalise the data
    '''
    for i in range(len(data_test)):    
        data_point = np.array([])
        data_point = data_test[i,:,:]
        data_point = data_point/np.max(data_point)
        data_test[i,:,:] = data_point
    
    return data_test

class MelspecDataset(Dataset):
    def __init__(self,somedata,label):
        self.data = somedata
        self.label = label
        
    def __len__(self):
        return len(self.data) #2452
 
    def __getitem__(self,idx):
        melspec = self.data[idx,:,:]
        lbl = self.label[idx]
        sample = (melspec,lbl)

        return sample

# load the data
data_train = np.load('train_data.npy')
# load the label
label_train = np.load('train_label.npy')

# ConvNet hyperparameters 
data_shape = data_train.shape[1:]
nb_pool = 2
nb_conv = 3
batch_size = 32
epochs = 50
learn_rate = 0.001
no_classes = 8 
validation_split = 0.1

# Assign class object
net = MyModel(nb_conv= nb_conv, nb_pool= nb_pool,no_classes = no_classes) # 3,2,8
net = net.double()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# normalise the train data
data_train = data_normalise(data_train)

# Split train data into train/validation sets
data_val = data_train[0:int(len(data_train)*validation_split),:,:]
data_train =  data_train[int(len(data_train)*validation_split):len(data_train),:,:]
label_val = label_train[0:int(len(label_train)*validation_split)]
label_train =  label_train[int(len(label_train)*validation_split):len(label_train)]

# After splitting, convert to Tensor and resize
data_val = (torch.from_numpy(data_val)).unsqueeze(1)
data_train = (torch.from_numpy(data_train)).unsqueeze(1)

train_dataset = MelspecDataset(data_train,label_train)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

val_dataset = MelspecDataset(data_val,label_val)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = learn_rate)


'''
for i_batch, sample_batched in enumerate(train_dataloader):
    print(sample_batched[0].size())
    print(sample_batched[0])
 '''

# Train the model
total_step_train = len(train_dataloader)
num_epoch = 50
loss_list = []
acc_list = []

net.train()
net = net.to(device)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_dataloader):
        #images = images.float()
        #labels = labels.float()
        images = images.cuda() #GPU 
        labels = labels.cuda() #GPU

        
        # Run the forward pass
        outputs = net(images)
        labels = labels.long()
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % total_step_train == 0:
        
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epoch, i + 1, total_step_train, loss.item(),(correct / total) * 100))

# Save model 
torch.save(net.state_dict(),'pytorch_model.ckpt')
print("Saved model to disk")

# For freezing the network except final layer
for param in net.parameters():
    param.requires_grad = False 

# Evaluate
total_step_train = len(val_dataloader)
valloss_list = []
valacc_list = []
# Set to eval mode. Required for pytorch.    
net.eval()
net = net.to(device)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(val_dataloader):
        #images = images.float()
        #labels = labels.float()
        images = images.cuda() #GPU 
        labels = labels.cuda() #GPU

        
        # Run the forward pass
        outputs = net(images)
        labels = labels.long()
        loss = criterion(outputs, labels)
        valloss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
    
        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        valacc_list.append(correct / total)

        if (i + 1) % total_step_train == 0:
        
            print('Epoch [{}/{}] of eval, Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epoch, i + 1, total_step_train, loss.item(),(correct / total) * 100))


# summarize history for accuracy
plt.plot(acc_list)
plt.plot(valacc_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(loss_list)
plt.plot(valloss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

'''
# ConvNet hyperparameters 
data_shape = data_train.shape[1:]
nb_pool = 2
nb_conv = 3
batch_size = 32
epochs = 50
learn_rate = 0.001

# import and train the model 
model = audio_emoti_cnnmodel(data_shape, nb_conv, nb_pool, no_classes)
adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer= adam,
              metrics=['accuracy'])
model.summary()

validation_split = 0.10
history = model.fit(data_train, label_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=validation_split)

# save the model 
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
'''
