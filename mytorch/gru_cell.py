import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
    
        
        
        rx = np.matmul(self.Wrx, self.x) + self.brx
        rh = np.matmul(self.Wrh, self.hidden) + self.brh
        
        zx = np.matmul(self.Wzx, self.x) + self.bzx
        zh = np.matmul(self.Wzh, self.hidden) + self.bzh
        
        nx = np.matmul(self.Wnx, self.x) + self.bnx
        self.nh = np.matmul(self.Wnh, self.hidden) + self.bnh
        
        self.r = self.r_act(rx+rh)
        
        self.z = self.z_act(zx+zh)
        
        self.n = self.h_act((nx+(self.r * self.nh)))
        
        h_t = ((1- self.z)*self.n) + (self.z *self.hidden)
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.
        
        

        return h_t
        
        #raise NotImplementedError

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        
    
        
        
        da_r = self.r_act.backward() 
        da_z = self.z_act.backward() 
        da_n = self.h_act.backward(state = self.n)
        
        
        hz = -self.n + self.hidden
        dz = np.multiply(delta,hz)
        
        hn= 1- self.z
        dn= np.multiply(delta, hn)
        
        act_n= da_n * dn
        act_z= da_z * dz
        
        dr = (act_n) * (self.nh)
        
        
        act_r= da_r * dr
        
        
        
        hh = self.z
        wnh_act = np.multiply(act_n,self.r)
        print(act_n.shape)
        print(self.r.shape)
        
        dh_prev_t = (delta * hh) + np.matmul(wnh_act,self.Wnh) + np.matmul(act_z,self.Wzh) + np.matmul(act_r,self.Wrh)
        
        dx = np.matmul(act_n,self.Wnx) + np.matmul(act_z,self.Wzx) + np.matmul(act_r,self.Wrx)
        
        shaped_x = self.x.reshape(1,self.d)
        
        self.dWnx += np.matmul(act_n.T,shaped_x)
        self.dWrx += np.matmul(act_r.T,shaped_x)
        self.dWzx += np.matmul(act_z.T,shaped_x)
        
        
        self.dbnx += act_n.reshape(-1)
        self.dbrx += act_r.reshape(-1)
        self.dbzx += act_z.reshape(-1)
        
        
        shaped_h = self.hidden.reshape(1,-1)
        
        print(shaped_h.shape)
        print(wnh_act.shape)
        
        self.dWnh += np.matmul(wnh_act.T,shaped_h)
        self.dWrh += np.matmul(act_r.T,shaped_h)
        self.dWzh += np.matmul(act_z.T,shaped_h)
        
        self.dbnh += wnh_act.reshape(-1)
        self.dbrh += act_r.reshape(-1)
        self.dbzh += act_z.reshape(-1)
        
        
        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)
        
        
        

        return dx, dh_prev_t
        #raise NotImplementedError
