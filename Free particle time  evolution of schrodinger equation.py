#Free particle time  evolution of schrodinger equation
#Crank Nicolson
import numpy as np
import matplotlib.pyplot as plt

L=40.0
N=1000 
a=2.0;  x0=-5.0; k=1.0   #wave function constants, a stands for sigma, x0 is mean, k is wave vector
xs, h= np.linspace(-L/2, L/2, N, retstep=True)  #Lattice
Vs= np.zeros_like(xs[1:-1])  #Free particle
#Vs=xs[1:-1]**2/2    #Harmonic oscillator
#In the array is xs negative if yes put in zero else 2 thus potential step
#Vs=np.where(xs[1:-1]<0, 0.,0.8)   #Barrier of adjustable height
us=np.exp(-(xs-x0)**2/a+(0+1.j)*k*xs)  #Gaussian function
dt=0.005
mu=dt*(0.+1.j)/(2*h**2)
t=0
tf=50.0
line =plt.plot(xs, np.abs(us)**2)[0]
#list object returned by matplotlib unpacked by using selecting [0]th element
#and storded in line object as it updates, as we want to store what is in the list
#but not just the list
plt.axvline(0)
plt.axhline(0)
plt.ylim(-.5, 1.1)
plt.plot(xs[1:-1], Vs)  #Potential plot 
#1/2 x**2 potential added
A1=np.eye(N-2)*(1+mu)+np.diag(1.j*dt/2*Vs)-mu/2*np.eye(N-2, k=1)-mu/2*np.eye(N-2, k=-1)
A2=np.eye(N-2)*(1-mu)+np.diag(-1.j*dt/2*Vs)+mu/2*np.eye(N-2, k=1)+mu/2*np.eye(N-2, k=-1)
B=np.dot(np.linalg.inv(A1), A2)

while t<tf:
      us[1:-1]=np.dot(B, us[1:-1])
      t+=dt
      line.set_ydata(np.abs(us)**2)
      plt.pause(0.00001)
plt.show()
