# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:47:27 2021

@author: HOME
"""
import numpy as np
import matplotlib.pyplot as plt
'''
print("####### Zadanie 2 #######")
print("#######")
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print("######################################################")

print("\n")
print("4.1.1")
print("#######")
A = np.array([[1, 2, 3], [7, 8, 9]])
print(A)
A = np.array([[1, 2, 3],
[7, 8, 9]])
print(A)
A = np.array([[1, 2, \
# po backslash’u nie moze byc zadnego znaku!
3],
[7, 8, 9]])
print(A)
print("######################################################")

print("\n")
print("4.1.2")
print("#######")
v = np.arange(1,7)
print(v,"\n")
v = np.arange(-2,7)
print(v,"\n")
v = np.arange(1,10,3)
print(v,"\n")
v = np.arange(1,10.1,3)
print(v,"\n")
v = np.arange(1,11,3)
print(v,"\n")
v = np.arange(1,2,0.1)
print(v,"\n")
print("#######  ######   #######   #######   #######")
v = np.linspace(1,3,4)
print(v)
v = np.linspace(1,10,4)
print(v)
print("#######  ######   #######   #######   #######")
X = np.ones((2,3))
Y = np.zeros((2,3,4))
Z = np.eye(2) 
#Z= np.eye(2,2) 
#Z= np.eye(2,3)
Q = np.random.rand(2,5) 
#Q=np.round(10*np.random.rand((3,3)))
print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q)
print("######################################################")

#print("\n")
#print("4.1.3")
#print("#######")
#U = np.block([[A], [X,Z]])
#print(U)
#blad w razmiarach macierzy
#print("######################################################")

print("\n")
print("4.1.4")
print("#######")
V = np.block([[
np.block([
np.block([[np.linspace(1,3,3)],
[np.zeros((2,3))]]) ,
np.ones((3,1))])
],
[np.array([100, 3, 1/2, 0.333])]] )
print(V)
print("######################################################")

print("\n")
print("4.2")
print("#######")
print( V[0,2] )
print("\n")
print( V[3,0] )
print("\n")
print( V[3,3] )
print("\n")
print( V[-1,-1] )
print("\n")
print( V[-4,-3] )
print("\n")
print( V[3,:] )
print("\n")
print( V[:,2] )
print("\n")
print( V[3,0:3] )
print("\n")
print( V[np.ix_([0,2,3],[0,-1])] )
print("\n")
print( V[3] )
print("######################################################")

print("\n")
print("4.3")
print("#######")
Q = np.delete(V, 2, 0)
print(Q)
print("\n")
Q = np.delete(V, 2, 1)
print(Q)
print("\n")
v = np.arange(1,7)
print( np.delete(v, 3, 0) )
print("######################################################")

print("\n")
print("4.4")
print("#######")
print(np.size(v))
print(np.shape(v))
print(np.size(V))
print(np.shape(V))
print("######################################################")

print("\n")
print("4.5.1")
print("#######")
A = np.array([[1, 0, 0],
[2, 3, -1],
[0, 7, 2]] )
B = np.array([[1, 2, 3],
[-1, 5, 2],
[2, 2, 2]] )
print( A+B )
print( A-B )
print( A+2 )
print( 2*A )
print("######################################################")

print("\n")
print("4.5.2")
print("#######")
MM1 = A@B
print(MM1)
MM2 = B@A
print(MM2)
print("######################################################")

print("\n")
print("4.5.3")
print("#######")
MT1 = A*B
print(MT1)
MT2 = B*A
print(MT2)
print("######################################################")

print("\n")
print("4.5.4")
print("#######")
DT1 = A/B
print(DT1)
print("######################################################")

print("\n")
print("4.5.5")
print("#######")
C = np.linalg.solve(A,MM1)
print(C) # porownaj z macierza B
x = np.ones((3,1))
b = A@x
y = np.linalg.solve(A,b)
print(y)
print("######################################################")

print("\n")
print("4.5.6")
print("#######")
PM = np.linalg.matrix_power(A,2) # por. A@A
PT = A**2 # por. A*A
print(PM)
print(PT)
print("######################################################")

print("\n")
print("4.5.7")
print("#######")
A.T # transpozycja
A.transpose()
A.conj().T # hermitowskie sprzezenie macierzy (dla m. zespolonych)
A.conj().transpose()
print(A)
print("######################################################")

print("\n")
print("4.6")
print("#######")
A == B
A != B
2 < A
A > B
A < B
A >= B
A <= B
np.logical_not(A)
np.logical_and(A, B)
np.logical_or(A, B)
np.logical_xor(A, B)

print( np.all(A) )
print( np.any(A) )
print("\n")
print( v > 4 )
print("\n")
print( np.logical_or(v>4, v<2))
print("\n")
print( np.nonzero(v>4) )
print("\n")
print( v[np.nonzero(v>4) ] )
print("######################################################")

print("\n")
print("4.7")
print("#######")
print(np.max(A))
print(np.min(A))
print("\n")
print(np.max(A,0))
print(np.max(A,1))
print("\n")
print( A.flatten() )
print( A.flatten('F') )
print("######################################################")

print("\n")
print("5.1")
print("#######")
import matplotlib.pyplot as plt
x = [1,2,3]
y = [4,6,5]
plt.plot(x,y)
plt.show()
print("######################################################")

print("\n")
print("5.1.1")
print("#######")
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y)
plt.show()
print("######################################################")

print("\n")
print("5.1.2")
print("#######")
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y)
plt.show()
print("######################################################")

print("\n")
print("5.1.3")
print("#######")
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y,'r:',linewidth=6)
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Nasz pierwszy wykres')
plt.grid(True)
plt.show()
print("######################################################")

print("\n")
print("5.1.4")
print("#######")
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
y = y1*y2
l1, = plt.plot(x,y,'b')
l2,l3 = plt.plot(x,y1,'r:',x,y2,'g')
plt.legend((l2,l3,l1),('dane y1','dane y2','y1*y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres')
plt.grid(True)
plt.show()
print("######################################################")
'''

print("\n")
print("####### Zadanie 3 #######")
print("#######")


A1 = np.array([np.linspace(1,5,5),np.linspace(5,1,5)])
A2 = np.zeros((3,2))
A3 = np.ones((2,3))
A3 = A3*2
A4 = np.linspace(-90,-70,3)
A5 = np.ones((5,1))
A5 = A5*10


A = np.block([[A3], [A4]])
A = np.block([A2,A])
A = np.block([[A1],[A]])
A = np.block([A,A5])

print(A)
print("######################################################")


print("\n")
print("####### Zadanie 4 #######")
print("#######")
import numpy as np
B = A[1]+A[3]
print(B)
print("######################################################")


print("\n")
print("####### Zadanie 5 #######")
print("#######")
C=np.max(A,0)
print(C)
print("######################################################")


print("\n")
print("####### Zadanie 6 #######")
print("#######")
D=np.delete(B,0)
print(D)
D=np.delete(D,len(D)-1)
print(D)
print("######################################################")


print("\n")
print("####### Zadanie 7 #######")
print("#######")
D[D==4]=0
print(D)
print("######################################################")

print("\n")
print("####### Zadanie 8 #######")
print("#######")
E=np.delete(C,(C==np.max(C)))
E=np.delete(E,(E==np.min(E)))
print(E)
print("######################################################")


print("\n")
print("####### Zadanie 9 #######")
print("#######")
ma=np.max(A)
print(ma)
mi=np.min(A)
print(mi)
print('max')
for i in range(len(A)):
    temp=A[i]
    if np.max(temp)==ma:
        print(temp)
print('min')
for i in range(len(A)):
    temp=A[i]
    if np.min(temp)==mi:
        print(temp)
print("######################################################")
print("\n")
print("####### Zadanie 10 #######")
print("#######")
print(D.flatten()*E.flatten())
print('\n')
n=len(D)
F1=[]
F2=[]
F=[]
for i in range (n):
    if i<(n-2):
        a=D[i+1]*E[i+2]
        F1.append(a)
    elif i<(n-1):
        a=D[i+1]*E[0]
        F1.append(a)
    else:
        a=D[0]*E[1]
        F1.append(a)
        
for i in range (n):
    if i<(n-3):
        a=D[n-1]*E[n-2]
        F2.append(a)
    elif i<(n-2):
        a=D[i-1]*E[n-1]
        F2.append(a)
    else:
        a=D[i-1]*E[i-2]
        F2.append(a)
#print (F1)
#print (F2)

for i in range (n):
    a=F1[i]-F2[i]
    F.append(a)
print('D=',D)
print('E=',E)
print (F)
print("######################################################")

print("\n")
print("####### Zadanie 11 #######")
print("#######")
def zd11():
    Q= np.round(10*np.random.rand(3,3))
    a=0
    for i in range(len(Q)):
        a=a+Q[i,i]
    return(Q,a)
print(zd11())
print("######################################################")

print("\n")
print("####### Zadanie 12 #######")
print("#######")
def zd12(n):
    Q= np.round(10*np.random.rand(n,n))
    for i in range(len(Q)):
        Q[i,i]=0
        Q[(len(Q)-1)-i,i]=0
            
    return(Q)

print(zd12(3))
print("######################################################")

print("\n")
print("####### Zadanie 13 #######")
print("#######")
def zd13(n):
    Q= np.round(10*np.random.rand(n,n))
    F=[]
    f=0
    for i in range(len(Q)):
        if (i%2)!=0:
         F=Q[i]
         for j in range(len(F)):
             f=f+F[j]
    return(Q,f)      
print(zd13(3))
print("######################################################")

print("\n")
print("####### Zadanie 14 #######")
print("#######")
f14 = lambda x: np.sin(2.0*x)
x = np.arange(-10.0, 10.0, 0.01)
y=f14(x)
plt.plot(x,y,'r--') 
print("######################################################")

print("\n")
print("####### Zadanie 15 #######")
print("#######")
import zad15

x = np.arange(-10,10,0.1)
y = zad15.lamda_15(x)
plt.plot(x,y,'g+')
print("######################################################")

print("\n")
print("####### Zadanie 17 #######")
print("#######")
import zad15

x = np.arange(-10.0, 10.0, 0.01)
y= 3*f14(x)+zad15.lamda_15(x)
plt.plot(x,y,'b*')
print("######################################################")

print("\n")
print("####### Zadanie 19 #######")
print("#######")

a = 1000000
x = np.linspace(0, 2*np.pi, a)
y = np.sin(x)
calka = np.sum(2*np.pi/a*y)
