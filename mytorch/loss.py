import numpy as np
import sys

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        z      = self.A - self.Y
        se     =  z ** 2
        Ones_C   = np.ones((C, 1), dtype="f")
        Ones_N   = np.ones((N, 1), dtype="f")
        sse_1 = np.dot(Ones_N.T,se)
        sse_= np.dot(sse_1,Ones_C)
        mse    = sse_/(N*C)
        
        return mse
    
    def backward(self):
    
        dLdA = self.A - self.Y
        
        return dLdA

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A   = A
        self.Y   = Y
        N        = A.shape[0]
        C        = A.shape[1]
        Ones_C   = np.ones((C, 1), dtype="f")
        Ones_N   = np.ones((N, 1), dtype="f")
        

        self.softmax     = np.exp(self.A)/(np.dot(np.dot(np.exp(self.A),Ones_C),Ones_C.T))
        crossentropy     = np.multiply(-self.Y,np.log(self.softmax)) 
        sce_1 = np.dot(Ones_N.T,crossentropy)
        sum_crossentropy = np.dot(sce_1,Ones_C) # TODO
        L = sum_crossentropy / N
        
        return L
    
    def backward(self):
    
        dLdA = self.softmax - self.Y # TODO
        
        return dLdA
