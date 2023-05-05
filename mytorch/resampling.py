import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        
        batch_size, in_channels, input_width = A.shape[0], A.shape[1], A.shape[2]
        
        self.input_width = input_width
        
        output_width = input_width * self.upsampling_factor - (self.upsampling_factor - 1)
        
        z = np.zeros((batch_size, in_channels, output_width))
        
        z[:,:,::self.upsampling_factor] = A[:,:,:]
        

        #Z = None # TODO
                                                                                                                          
        return z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        batch_size, in_channels, output_width = dLdZ.shape[0], dLdZ.shape[1], dLdZ.shape[2]
        
        dLdA = np.zeros((batch_size, in_channels, self.input_width))
        
        dLdA[:,:,:] = dLdZ[:,:,::self.upsampling_factor]
        
        #dLdA = None  #TODO

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        
        self.init_size = A.shape[2]
        

        Z = A[:,:,::self.downsampling_factor] # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        batch_size, in_channels, output_width = dLdZ.shape[0], dLdZ.shape[1], dLdZ.shape[2]
        
        dLdA = np.zeros((batch_size, in_channels, self.init_size)) #TODO
        
        dLdA[:,:,::self.downsampling_factor] = dLdZ[:,:,:]

        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        
        batch_size, in_channels, input_width, input_height = A.shape[0], A.shape[1], A.shape[2], A.shape[3]
        
        self.input_width = input_width
        
        self.input_height = input_height
        
        output_width = input_width * self.upsampling_factor - (self.upsampling_factor - 1)
        
        output_height = input_height * self.upsampling_factor - (self.upsampling_factor - 1)
        
        z = np.zeros((batch_size, in_channels, output_width, output_height))
        
        z[:,:,::self.upsampling_factor,::self.upsampling_factor] = A[:,:,:,:]

       # Z = None # TODO

        return z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        batch_size, in_channels, output_width, output_height = dLdZ.shape[0], dLdZ.shape[1], dLdZ.shape[2], dLdZ.shape[3]
        
        dLdA = np.zeros((batch_size, in_channels, self.input_width, self.input_height))
        
        dLdA[:,:,:,:] = dLdZ[:,:,::self.upsampling_factor,::self.upsampling_factor]

        #dLdA = None  #TODO

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        
        self.input_width = A.shape[2]
        
        self.input_height = A.shape[3]
        
        Z = A[:,:,::self.downsampling_factor,::self.downsampling_factor] # TODO

        #Z = None # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        batch_size, in_channels, output_width, output_height = dLdZ.shape[0], dLdZ.shape[1], dLdZ.shape[2], dLdZ.shape[3]
        
        dLdA = np.zeros((batch_size, in_channels, self.input_width, self.input_height)) #TODO
        
        dLdA[:,:,::self.downsampling_factor,::self.downsampling_factor] = dLdZ[:,:,:,:]
        
        
        #dLdA = None  #TODO

        return dLdA
    
    
    
    
    
    
    
    
    
    