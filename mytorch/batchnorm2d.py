# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        # num features: number of channels
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        if eval:
            self.Z         = Z
            self.N         = Z.shape[0] # TODO
            
            ones           = np.ones((1,self.N))
            
            # self.M         = np.mean(self.Z,axis=(0,2,3),keepdims=True) # TODO
            # #norm           = np.matmul(ones.T,self.running_M)
            # norm_1         = self.Z - self.M
            # self.V         = np.var(self.Z,axis=(0,2,3),keepdims=True)
            # NZ_denom       = np.sqrt(self.V + self.eps)
            # self.NZ        = norm_1 / NZ_denom # TODO
            # self.BZ        = np.multiply(self.BW,self.NZ) + self.Bb # TODO
            #self.Z = Z
            self.NZ = (self.Z-self.running_M)/np.sqrt(self.running_V+self.eps)
            self.BZ = (self.BW*self.NZ)+self.Bb
            #return BZeval
            return self.BZ

        self.Z = Z
        self.N = Z.shape[0]  # TODO
        
        self.M = np.mean(self.Z,axis=(0,2,3),keepdims=True)  # TODO
        self.V = np.var(self.Z,axis=(0,2,3),keepdims=True)
        
        #self.M         = np.matmul(ones,self.Z)/self.N # TODO
        #norm           = np.matmul(ones.T,self.M)
        norm_1         = self.Z - self.M
        #self.V         = np.matmul(ones,(norm_1)**2)/ self.N # TODO
        NZ_denom       = np.sqrt(self.V + self.eps)
        self.NZ        = norm_1 / NZ_denom # TODO
        #self.NZ        =(self.Z - self.M)/np.sqrt(self.V + self.eps)
        self.BZ        = np.multiply(self.BW,self.NZ) + self.Bb # TODO
        

        self.running_M =  self.alpha * self.running_M + (1 - self.alpha) * self.M   # TODO
        self.running_V =  self.alpha * self.running_V + (1 - self.alpha) * self.V  # TODO

        return self.BZ

    def backward(self, dLdBZ):
        
        self.dLdBb =  np.sum(dLdBZ, axis=(0, 2, 3), keepdims=True)  # TODO (0, 2, 3), keepdims=True
        self.dLdBW = np.sum(np.multiply(dLdBZ,self.NZ), axis=(0, 2, 3), keepdims=True) # TODO
        

        dLdNZ = np.multiply(dLdBZ,self.BW)  # TODO
        n = self.Z - self.M
        sigma = self.V + self.eps

        dLdV = np.sum((np.multiply(np.multiply(dLdNZ,n),sigma**(-3/2))),axis=(0, 2, 3), keepdims=True)/-2  # TODO
        dLdM = -np.sum(np.multiply(dLdNZ, sigma**(-0.5)),axis=(0, 2, 3), keepdims=True)- ((2/self.N*self.Z.shape[2]*self.Z.shape[3])* (np.multiply(dLdV,np.sum(n,axis=(0, 2, 3), keepdims=True))))  # TODO

        dLdZ = (dLdM/(self.N*self.Z.shape[2]*self.Z.shape[3])) + np.multiply(dLdNZ,(sigma**(-0.5))) + np.multiply(dLdV ,n*(2/self.N*self.Z.shape[2]*self.Z.shape[3]))  # TODO
        

        return  dLdZ
