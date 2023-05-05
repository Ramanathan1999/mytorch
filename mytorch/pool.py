import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        self.batch_size, self.in_channels, self.input_width, self.input_height = A.shape
        self.output_width, self.output_height = (self.input_width-self.kernel+1), (self.input_height-self.kernel+1)
        
        Z = np.zeros((self.batch_size, self.in_channels, self.output_width, self.output_height))
        self.index =  np.zeros((self.batch_size, self.in_channels, self.output_width, self.output_height))
        
        
        for b in range(self.batch_size):
            for c in range(self.in_channels):
                for i in range(self.output_width):
                    for j in range(self.output_height):
                        start_width = i
                        end_width = i + self.kernel
                        start_height = j
                        end_height = j+self.kernel
                        Z[:,:,i,j]= np.max(A[:,:,start_width:end_width,start_height:end_height], axis=(2,3))
                        self.index[b,c,i,j]= np.argmax(A[b,c,start_width:end_width,start_height:end_height])
        
        
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros((self.batch_size, self.in_channels, self.input_width, self.input_height))
        
        for b in range(self.batch_size):
            for c in range(self.in_channels):
                for i in range(self.output_width):
                    for j in range(self.output_height):
                        index = self.index[b,c,i,j]
                        x, y = np.unravel_index(int(index), (self.kernel, self.kernel))
                        dLdA[b,c,i+x,j+y] += dLdZ[b,c, i, j]
                        
        
        # for i in range(self.output_width):
        #     for j in range(self.output_height):
        #         index= self.index[:,:,i,j]
        #         x,y = index[0], index[1]
        #         value = dLdZ[:,:,i,j]
        #         dLdA[:,:,x,y]+= value
    
        return dLdA
    
class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        self.batch_size, self.in_channels, self.input_width, self.input_height = A.shape
        self.output_width, self.output_height = (self.input_width-self.kernel+1), (self.input_height-self.kernel+1)
        
        Z = np.zeros((self.batch_size, self.in_channels, self.output_width, self.output_height))
        
        for i in range(self.output_width):
            for j in range(self.output_height):
                start_width = i
                end_width = i + self.kernel
                start_height = j
                end_height = j+self.kernel
                Z[:,:,i,j]= np.mean(A[:,:,start_width:end_width,start_height:end_height], axis=(2,3))
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdA = np.zeros((self.batch_size, self.in_channels, self.input_width, self.input_height))
        for b in range(self.batch_size):
            for c in range(self.in_channels):
                for i in range(self.output_width):
                    for j in range(self.output_height):
                        mean= dLdZ[b,c,i,j]/((self.kernel)**2)
                        mask = np.ones((self.kernel,self.kernel))*mean
                        dLdA[b,c,i:i+(self.kernel),j:j+(self.kernel)]+= mask
        
        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)#TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        output = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(output)
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        output = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(output)
        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel) #TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        output = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(output)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        output = self.downsample2d.backward(dLdZ)
        dLdZ = self.meanpool2d_stride1.backward(output)
        return dLdZ