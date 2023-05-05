import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        
        self.alpha     = alpha
        self.eps       = 1e-8
        
        self.Z         = None
        self.NZ        = None
        self.BZ        = None

        self.BW        = np.ones((1, num_features))
        self.Bb        = np.zeros((1, num_features))
        self.dLdBW     = np.zeros((1, num_features))
        self.dLdBb     = np.zeros((1, num_features))
        
        self.M         = np.zeros((1, num_features))
        self.V         = np.ones((1, num_features))
        
        # inference parameters
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

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
            
            #self.M         = np.matmul(ones,self.Z)/self.N # TODO
            norm           = np.matmul(ones.T,self.running_M)
            norm_1         = self.Z - norm
            #self.V         = np.matmul(ones,(norm_1)**2)/ self.N # TODO
            NZ_denom       = np.matmul(ones.T, np.sqrt(self.running_V + self.eps))
            self.NZ        = norm_1 / NZ_denom # TODO
            self.BZ        = np.multiply(np.matmul(ones.T,self.BW),self.NZ) + np.matmul(ones.T,self.Bb) # TODO
            return self.BZ
            
        self.Z         = Z
        self.N         = Z.shape[0] # TODO
        
        ones           = np.ones((1,self.N))
        
        self.M         = np.matmul(ones,self.Z)/self.N # TODO
        norm           = np.matmul(ones.T,self.M)
        norm_1         = self.Z - norm
        self.V         = np.matmul(ones,(norm_1)**2)/ self.N # TODO
        NZ_denom       = np.matmul(ones.T, np.sqrt(self.V + self.eps))
        self.NZ        = norm_1 / NZ_denom # TODO
        self.BZ        = np.multiply(np.matmul(ones.T,self.BW),self.NZ) + np.matmul(ones.T,self.Bb) # TODO
        
        self.running_M = self.alpha * self.running_M + (1-self.alpha) * self.M # TODO
        self.running_V = self.alpha * self.running_V + (1-self.alpha) * self.V # TODO
        
        return self.BZ

    def backward(self, dLdBZ):
        
        self.dLdBb  = np.sum(dLdBZ, axis=0) # TODO
        self.dLdBW  = np.sum(np.multiply(dLdBZ,self.NZ), axis=0)# TODO
        
        dLdNZ       = np.multiply(dLdBZ,self.BW) # TODO
        
        #print(dLdNZ)
        
        n = self.Z - self.M
        sigma = self.V + self.eps
 # TODO
        dLdV        =  np.sum((np.multiply(np.multiply(dLdNZ,n),sigma**(-3/2))),axis=0)/-2 # TODO
        
       # print(dLdV)
        
        dLdM        = -np.sum(np.multiply(dLdNZ, sigma**(-0.5)),axis=0)- ((2/self.N)* (np.multiply(dLdV,np.sum(n,axis=0))))
        
       # print(dLdM)
        
        dLdZ        = dLdM/self.N + np.multiply(dLdNZ,(sigma**-0.5)) + np.multiply(dLdV ,(self.Z-self.M)*(2/self.N)) # TODO

        #print(dLdZ)
        return  dLdZ
    
    
    
    
    
    
    
    