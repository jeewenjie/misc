#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:49:19 2019

@author: pratik
"""
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
from pytorch_cnn_model import MyModel
from torch.utils.data import DataLoader,Dataset
import torch 
import torch.optim as optim
#from torchsummary import summary

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

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

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

# load the test data
data_test = np.load('test_data.npy')
# load the test label
label_test = np.load('test_label.npy')

# Define classes
class_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

nb_conv = 3
nb_pool = 2
no_classes =8
batch_size = 32
# Load model
net = MyModel(nb_conv,nb_pool,no_classes)#######################################
net = net.cuda()
net = net.double()

net.load_state_dict(torch.load('pytorch_model.ckpt')) #,map_location ='cpu')) # Remove map_location argument on gpu machine
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#summary(net, input_size=(1, 128, 400))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001,betas=(0.9,0.999),weight_decay=0,amsgrad=False)

no_classes = 8 

data_test = data_normalise(data_test)

data_test = (torch.from_numpy(data_test)).unsqueeze(1)

test_dataset = MelspecDataset(data_test,label_test)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
prediction_array = np.array([])
loss_list = []
acc_list = []
#print(label_test)

'''
for i_batch, sample_batched in enumerate(test_dataloader):
    print(sample_batched[1].size())
    print(sample_batched[1])
'''
for param in net.parameters():
    param.requires_grad = False 

net.eval() # Set to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0 
    for i, (inputs, labels) in enumerate(test_dataloader):
        inputs = inputs.cuda() #GPU 
        labels = labels.cuda() #GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        outputs = net(inputs)
        labels = labels.long()
        #loss = criterion(outputs, labels)
        #loss_list.append(loss.item())
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        prediction_array = np.concatenate((prediction_array,preds),axis=None)
        
    print('Test Accuracy of the model: {} %'.format((correct / total) * 100))

#print(prediction_array)

matrix = confusion_matrix(label_test,prediction_array)
#matrix = confusion_matrix(label_test.argmax(axis=1), y_prediction.argmax(axis=1))


plt.figure()
plot_confusion_matrix(matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()

'''
# load model and model's weights
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# runthe model on the test data
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
loaded_model.compile(loss='categorical_crossentropy',
              optimizer= adam,
              metrics=['accuracy'])
score = loaded_model.evaluate(data_test, label_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# print the confusion matrix
y_prediction = loaded_model.predict(data_test)
matrix = confusion_matrix(label_test.argmax(axis=1), y_prediction.argmax(axis=1))
class_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

plt.figure()
plot_confusion_matrix(matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()

'''