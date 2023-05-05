# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        #print(self.A)
        #print(self.A.shape)
        #print(type(self.A))
        
        batch_size, in_channels, self.input_size = A.shape
        
        output_size = self.input_size - self.kernel_size + 1
        
        Z= np.zeros((self.A.shape[0],self.out_channels,output_size))
        
        for i in range(output_size):
            start = i
            end = i+ self.kernel_size
            Z[:,:,i]= (np.tensordot(self.A[:,:,start:end],self.W, axes=([1, 2], [1, 2]))) +self.b  

        #Z = None # TODO
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        
        
        batch_size, out_channels, output_size = dLdZ.shape[0], dLdZ.shape[1], dLdZ.shape[2]
        
        
        #dLdW
        #zero = np.zeros((1,self.A.shape[1]))
        
        #zero_new = zero[:,np.newaxis]
        
        #broadcasted_dLdZ = zero_new + dLdZ
        
        #steps_w = self.A.shape[2]- self.kernel_size +1
        
        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size))
        
        for i in range(self.kernel_size):
            start , end = i, i+ output_size
            #self.dLdW[:,:,i] = np.sum((broadcasted_dLdZ*self.A[:,:,start:end]),axis=1)
            self.dLdW[:,:,i] = np.tensordot(dLdZ, self.A[:,:,start:end], axes=([0, 2],[0, 2]))
            
        #dLdA
        flipped_weight = np.flip(self.W,axis=2)
        
        padding = self.kernel_size - 1
        padded = np.pad(dLdZ,((0,0),(0,0),(padding,padding)),constant_values='0')
        
        #steps_a = padded.shape[2] - flipped_weight.shape[2] +1
        dLdA = np.zeros((batch_size, self.in_channels, self.input_size))
        
        for j in range(self.input_size):
            start, end = j, j + self.kernel_size
            dLdA[:,:,j]= np.tensordot(padded[:,:,start:end],flipped_weight,axes=([1,2],[0,2]))
        
        
        #self.dLdW =  # TODO        
        self.dLdb = np.sum(dLdZ, axis=(0,2)) # TODO
        #dLdA = None # TODO

        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
    
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None) # TODO
        self.downsample1d = Downsample1d(downsampling_factor=stride) # TODO Downsample1d()

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1 
        A = self.conv1d_stride1.forward(A)
        # TODO

        # downsample
        Z = self.downsample1d.forward(A) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ) # TODO 

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        
        batch_size, self.in_channels, self.input_width, self.input_height = self.A.shape[0], self.A.shape[1], self.A.shape[2], self.A.shape[3]
        
        output_width = self.input_width - self.kernel_size + 1
        output_height = self.input_height - self.kernel_size + 1
        
        Z= np.zeros((batch_size,self.out_channels,output_width,output_height))
        
        for j in range(output_width):
            for i in range(output_height):
                start_width = i
                end_width = i+ self.kernel_size
                start_height = j
                end_height = j + self.kernel_size
                
                Z[:,:,i,j]= (np.tensordot(self.A[:,:,start_width:end_width,start_height:end_height],self.W,axes=([1,2,3], [1,2,3]))) +self.b 

        #Z = None #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        
        batch_size, out_channels, output_width, output_height = dLdZ.shape[0], dLdZ.shape[1], dLdZ.shape[2], dLdZ.shape[3]
        
        
        #dLdW
        #self.dLdW = np.zeros((batch_size, in_channels,steps_w,steps_w))
        
        
        steps_w = dLdZ.shape[2]- self.kernel_size +1
        steps_h = dLdZ.shape[3]- self.kernel_size +1
        
        for j in range(self.kernel_size):
            for k in range(self.kernel_size):
            
                start_width, end_width, start_height, end_height = j, j + output_width,k, k + output_height
                self.dLdW[:,:,j,k] = np.tensordot(dLdZ,self.A[:,:,start_width:end_width, start_height:end_height],axes=([0,2,3],[0,2,3]))
            
        #dLdA
        flipped_weight = np.flip(self.W, (2,3))
        padding = self.kernel_size - 1
        padded = np.pad(dLdZ,((0, 0), (0, 0),(padding,padding),(padding,padding)),constant_values='0')
        
        #steps_a = padded.shape[1] - flipped_weight.shape[2] +1
        
        dLdA = np.zeros((batch_size, self.in_channels,self.input_width,self.input_height))
        
        for j in range(self.input_width):
            for k in range(self.input_height):
            
                  start_width, end_width, start_height, end_height = j, j + self.kernel_size,k, k + self.kernel_size
                  dLdA[:,:,j,k]= np.tensordot(padded[:,:,start_width:end_width, start_height:end_height],flipped_weight, axes=([1,2,3], [0, 2,3]))
        
        #self.dLdW = None # TODO    
        
        self.dLdb = np.sum(dLdZ, axis=(0,2,3)) # TODO
        #dLdA = None # TODO

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels,
                     kernel_size, weight_init_fn=None, bias_init_fn=None) # TODO
        self.downsample2d = Downsample2d(downsampling_factor=stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        # TODO

        # downsample
        #Z = None # TODO
        
        A = self.conv2d_stride1.forward(A)
        # TODO

        # downsample
        Z = self.downsample2d.forward(A) # TODO

        return Z

        #return NotImplemented

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        # TODO

        # Call Conv1d_stride1 backward
        #dLdA = None # TODO 
        
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ) # TODO 

        return dLdA
        

        #return NotImplemented

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        #TODO
        self.upsample1d = Upsample1d(upsampling_factor) #TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        #TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A) #TODO

        # Call Conv1d_stride1()
        Z =  self.conv1d_stride1.forward(A_upsampled)  #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #TODO

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ) #TODO

        dLdA =  self.upsample1d.backward(delta_out) #TODO

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) #TODO
        self.upsample2d = Upsample2d(upsampling_factor)  #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A) #TODO

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled) #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ) #TODO

        dLdA =  self.upsample2d.backward(delta_out) #TODO

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        
        self.batch, self.channel, self.weight = A.shape
        Z = np.reshape(A, (self.batch, self.channel * self.weight))

        #Z = None # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = np.reshape(dLdZ, (self.batch, self.channel, self.weight)) #TODO

        return dLdA

