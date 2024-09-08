import numpy as np
import torch
import torch.nn as nn
def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 000000000

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        n = 3           # 3 output channels for conv1
        kernel_size = 5  # 5x5 kernel
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n,kernel_size=kernel_size,padding=padding)
        # TODO: complete this method
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=n*2, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=n*2, out_channels=n*4, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=n*4, out_channels=n*8, kernel_size=kernel_size, padding=padding)
        # self.fc1 = nn.Linear(in_features=8*n, out_features=100) # fully connected 100 hidden units
        self.fc1 = nn.Linear(in_features=8*n*392, out_features=100) # fully connected 100 hidden units  8*n*(224//16)*(448//16)
        self.fc2 = nn.Linear(in_features=100, out_features=2)   # fully connected 2 hidden units
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # N,3,448,224
        out = self.conv1(inp)   # Layer 1 -> N, 32, 448, 224
        out = self.relu(out)
        out = self.maxpool(out) # -> N, 32, 224, 112
        
        out = self.conv2(out)   # Layer 2 -> N, 64, 224, 112
        out = self.relu(out)
        out = self.maxpool(out) # -> N, 64, 112, 56
        
        out = self.conv3(out)   # Layer 3 -> N, 128, 112, 56
        out = self.relu(out)
        out = self.maxpool(out) # -> N, 128, 56, 28
        
        out = self.conv4(out)   # Layer 4 -> N, 256, 56, 28
        out = self.relu(out)
        out = self.maxpool(out) # -> N, 256, 28, 14
        
        out = out.reshape(out.size(0), -1)  # Flatten the feature maps using reshape -> N, 100352
        out = self.relu(self.fc1(out)) # -> N, 100
        out = self.fc2(out)            # -> N, 2
        return out

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        # TODO: complete this method
        n = 8            # 8 output channels for conv1   
        kernel_size = 3  # 3x3 kernel
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=6,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=n*2, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=n*2, out_channels=n*4, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=n*4, out_channels=n*8, kernel_size=kernel_size, padding=padding)
        self.fc1 = nn.Linear(in_features=8*n*196, out_features=100) #8*n*(224//16)*(224//16)
        self.fc2 = nn.Linear(in_features=100, out_features=2)  # fully connected 100 and 2 hidden units
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    # TODO: complete this class
    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # TODO start by changing the shape of the input to (N,6,224,224)
        # TODO: complete this function
        # # Concatenate the left and right shoe images along the channel dimension
        image1 = inp[:, :, :224, :]
        image2 = inp[:, :, 224:, :]
        
        # Concatenate the images along the channel axis (second dimension)
        inp_concat = torch.cat((image1, image2), dim=1)
        
        out = self.conv1(inp_concat)    # Layer 1
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.conv2(out)           # Layer 2
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.conv3(out)           # Layer 3
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.conv4(out)           # Layer 4
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = out.reshape(out.size(0), -1)  # Flatten the feature maps using reshape
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
