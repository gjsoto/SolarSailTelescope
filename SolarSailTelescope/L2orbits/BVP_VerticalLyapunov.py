from CR3BP import CR3BP
import numpy as np
import pylab as pl
import astropy as astro
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import astropy.constants as const
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.interpolate as interpolate
import cPickle as pickle
from scipy.integrate import solve_bvp
from astropy.coordinates import SkyCoord
import scipy.optimize as optimize
import scipy.integrate as itg
from copy import deepcopy
import time

##################################################

class BVP_VerticalLyapunov(CR3BP):
    
    def __init__(self,**specs):
        CR3BP.__init__(self,**specs) 
        print 'getting Vertical Lyapunovs'
    
    def linearVLApprox(self,Ax,Az,t):
        
        c2 = self.c_fun(2).value
        
        #Calculating frequencies and other constants
        #for the Richardson Third Order Approximation
        wp =  0.5*(c2-2-np.sqrt(9*c2**2-8*c2))
        wp = np.sqrt(-wp)
        
        wv = np.sqrt(c2)
        
        k= (wp**2 + 1 + 2*c2)/(2*wp)
        
        #conversion factor
        Ax = Ax.to('au').value/self.g2
        Az = Az.to('au').value/self.g2
        
        x = -Ax*np.cos(wp*t)
        y = k*Ax*np.sin(wp*t)
        z = Az*np.sin(wv*t)
        dx = wp*Ax*np.sin(wp*t)
        dy = wp*k*Ax*np.cos(wp*t)
        dz = wv*Az*np.cos(wv*t)
        
        T = 2*np.pi*(1/wv)
        
        x = self.g2*x + 1 - self.mu + self.g2
        y = self.g2*y
        z = self.g2*z
        
        dx = self.g2*dx
        dy = self.g2*dy
        dz = self.g2*dz
        
        return x,y,z,dx,dy,dz,T
    
    def bc_VL(self,sA,sB):
        """Creates boundary conditions for solving a boundary value problem
        
        This method returns the boundary conditions for the starshade transfer
        trajectory between the lines of sight of two different stars. Point A
        corresponds to the starshade alignment with star A; Point B, with star B.
        
        Args:
            rA (float 1x3 ndarray):
                Starshade position vector aligned with current star of interest
            rB (float 1x3 ndarray):
                Starshade position vector aligned with next star of interest
                
        Returns:
            BC (float 1x6 ndarray):
                Star position vector in rotating frame in units of AU
        """
        BC1 = sA[1]
        BC2 = sB[1]
        
        BC3 = sB[2]
        
        BC4 = sA[3]
        BC5 = sB[3]
        
        BC6 = sA[5]
        BC7 = sB[5] - self.dz0
        
        BC = np.array([BC1,BC2,BC3,BC4,BC5,BC6,BC7])
        
        return BC
         
    def silent_cartographer(self,Ax,Az,Fam):

        #initial and final times (quarterway through full orbit)
        tau = np.linspace(0,0.25,2)
        
        x  = []
        y  = []
        z  = []
        dx = []
        dy = []
        dz = []
        T  = []
        
        for i in range(Fam):
            # first guess
            xG,yG,zG,dxG,dyG,dzG,TG = self.linearVLApprox(Ax,Az,0)
            sGa = np.array([xG,yG,zG,dxG,dyG,dzG,TG])
            
            #invariant parameter
            self.dz0 = dzG.copy()
            
            #final guess
            xG,yG,zG,dxG,dyG,dzG,TG = self.linearVLApprox(Ax,Az,0.25*np.pi)
            sGb = np.array([xG,yG,zG,dxG,dyG,dzG,TG])
            
            sG = np.vstack([sGa,sGb]).reshape(2,7)
            sol = solve_bvp(self.equationsOfMotion_T,self.bc_VL,tau,sG.T,tol=1e-10)
            if sol.status != 0:
                print("WARNING: BVP sol.status is %d" % sol.status)
            print(sol.message)
            
            s = sol.y
#            s,Tp = self.refineVL_diffCorrect(sol.y[:,0])
            
            x.append(s[0][0])
            y.append(s[1][0])
            z.append(s[2][0])
            dx.append(s[3][0])
            dy.append(s[4][0])
            dz.append(s[5][0])
            T.append(s[6][0])
            
            Ax += 30000*u.km
            Az += 30000*u.km
        
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        dx = np.array(dx)
        dy = np.array(dy)
        dz = np.array(dz)
        T = np.array(T)
        
        return x,y,z,dx,dy,dz,T
    
    def refineVL_diffCorrect(self,s0):
#        import pdb
        x,y,z,dx,dy,dz,T = s0
        
        sJ0 = np.array([x,y,z,dx,dy,dz])
        I6 = np.eye(6)
        sJ0 = np.hstack([sJ0,I6.flatten()])
        
        IAintEverLeft = lambda t,s: s[1] if t>1e-1 and s[2]<0 and s[4]>0 else 100
        
        IAintEverLeft.terminal  = 1
        IAintEverLeft.direction = 1
        
        sJ,te = self.integrate(sJ0,[0,3*T],IAintEverLeft,None,True)
#        inits = np.array([0,2,4])
        x,y,z,dx,dy,dz = sJ[0:6,-1]
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        ax.plot(sJ[0,:],sJ[1,:],sJ[2,:])
        ax.plot([sJ[0,0]],[sJ[1,0]],[sJ[2,0]],'x',label='Start')
        ax.plot([sJ[0,-1]],[sJ[1,-1]],[sJ[2,-1]],'d',label='Finish')
        print te
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        
        
        
        for i in range(8):
            
            ax.plot(sJ[0,:],sJ[1,:],sJ[2,:],label='Loop %d' % i)
            
            
            dx,dy,dz,ddx,ddy,ddz = self.equationsOfMotion(te,sJ[0:6,-1])
            Phi = sJ[:,-1][6:].reshape(6,6)
            
            P = np.array([[Phi[3,2],Phi[3,4]],[Phi[5,2],Phi[5,4]]])
            
            ddots = np.array([ddx,ddz])
            Phiy  = np.array([Phi[1,2],Phi[1,4]])
            
            P -= (1/dy) * np.tensordot(ddots,Phiy,axes=0)
            
            deltaZ,deltaDY = np.dot( np.linalg.inv(P) , -np.array([dx,dz]))
            sJ0[0:6] += np.array([0 , 0 , deltaZ , 0 , deltaDY , 0])
            
#            import pdb
#            pdb.set_trace()
            
            
            sJ,te = self.integrate(sJ0,[0,3*T],IAintEverLeft,None,True)
            x,y,z,dx,dy,dz = sJ[0:6,-1]
            
            print 'Loop No.',i
            print sJ0[0:6]
            print sJ[0:6,-1]
            
            if np.abs(y) < 1e-14 and np.abs(dx) < 1e-14 and np.abs(dz) < 1e-14:
                break
        ax.legend()    
        return sJ,T
    
