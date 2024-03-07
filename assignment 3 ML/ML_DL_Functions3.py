import numpy as np
import torch
import torch.nn as nn
def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 319149878

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        n = 32           # 32 output channels for conv1
        kernel_size = 3  # 3x3 kernel
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n,kernel_size=kernel_size,padding=padding)
        # TODO: complete this method
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=n*2, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=n*2, out_channels=n*4, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=n*4, out_channels=n*8, kernel_size=kernel_size, padding=padding)
        # self.fc1 = nn.Linear(in_features=8*n, out_features=100) # fully connected 100 hidden units
        self.fc1 = nn.Linear(in_features=100352, out_features=100) # fully connected 100 hidden units  8*n*(224//16)*(448//16)
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
        out = self.conv1(inp)   # Layer 1
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.conv2(out)   # Layer 2
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.conv3(out)   # Layer 3
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.conv4(out)   # Layer 4
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = out.reshape(out.size(0), -1)  # Flatten the feature maps using reshape
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        # TODO: complete this method
        n = 32           # 32 output channels for conv1
        kernel_size = 3  # 3x3 kernel
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=6,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=n*2, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=n*2, out_channels=n*4, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=n*4, out_channels=n*8, kernel_size=kernel_size, padding=padding)
        self.fc1 = nn.Linear(in_features=93184, out_features=100) #8*n*(224//16)*(448//16)
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
        # Concatenate the left and right shoe images along the channel dimension
        left_shoe = inp[:, :, :, :3]  # Select the first 3 channels for left shoe
        right_shoe = inp[:, :, :, 3:]  # Select the last 3 channels for right shoe
        # Resize images if they don't have the same dimensions
        if left_shoe.shape[2:] != right_shoe.shape[2:]:
            left_shoe = nn.functional.interpolate(left_shoe, size=right_shoe.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate the left and right shoe images along the channel dimension
        inp_concat = torch.cat((left_shoe, right_shoe), dim=1)
        
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