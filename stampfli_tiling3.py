#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:35:03 2019

@author: babreu
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

hsq3 = 0.5*np.sqrt(3.0)

#%%

vertices = []

## TRIANGLES

# T1
vertex = [0,0]
vertices.append(vertex)
vertex = [0,1]
vertices.append(vertex)
vertex = [0.5*np.sqrt(3.), 0.5]
vertices.append(vertex)
## T2
vertex = [0.5*np.sqrt(3.), -0.5]
vertices.append(vertex)
## T3
vertex = [0,-1]
vertices.append(vertex)
## T4 
vertex = [-0.5*np.sqrt(3.), -0.5]
vertices.append(vertex)
## T5
vertex = [-0.5*np.sqrt(3.), 0.5]
vertices.append(vertex)
## T6

#now save it this
hexagon = np.array(vertices[1:])


## T7
vertex = [0.5*(np.sqrt(3.)+1.), 0.5*(np.sqrt(3.)+1.)]
vertices.append(vertex)
vertex = [0.5*np.sqrt(3.) + 1., 0.5]
vertices.append(vertex)
## T8
vertex = [0.5*np.sqrt(3.)+1.0, -0.5]
vertices.append(vertex)
vertex = [0.5*(np.sqrt(3.)+1.), -0.5-0.5*np.sqrt(3)]
vertices.append(vertex)
## T9
vertex = [0.5, -1 - hsq3]
vertices.append(vertex)
vertex = [-0.5, -1 - hsq3]
vertices.append(vertex)
## T10
vertex = [-hsq3-0.5, -0.5-hsq3]
vertices.append(vertex)
vertex = [-hsq3-1., -0.5]
vertices.append(vertex)
## T11
vertex = [-hsq3-1., 0.5]
vertices.append(vertex)
vertex = [-hsq3-0.5, 0.5+hsq3]
vertices.append(vertex)
## T12
vertex = [-0.5,1+hsq3]
vertices.append(vertex)
vertex = [0.5,1+hsq3]
vertices.append(vertex)




#%%  
#######   FIRST INFLATION

lamb = 2. + np.sqrt(3.)

v = lamb*np.array(vertices)


#
def rotate_dodeca(x):
 
    theta = np.radians(30.0)
    c, s = np.cos(theta), np.sin(theta)
    x1 = c*x[0] + s*x[1]
    x2 = -s*x[0] + c*x[1]
    
    return [x1,x2]


## NOW LET'S BUILD DODECAGONS UP
vertices2 = []

theta = np.radians(15.0)
s=np.sin(theta)
t=np.tan(theta)


vertex = [0.5/t,0.5]
vertices2.append(vertex)
for i in range(11):
    vertex = rotate_dodeca(vertex)
    vertices2.append(vertex)

temp = list(vertices2)

for j in range(19):
    for ve in temp:
        vertex = [ve[0] + v[j,0], ve[1] + v[j,1]]
        vertices2.append(vertex)


### FILL THEM WITH HEXAGONS
hexagon2 = []
for i in range(6):
    x = [hexagon[i,0], hexagon[i,1]]
    x = rotate_dodeca(x)
    hexagon2.append(x) 
hexagon2 = np.array(hexagon2)       
        
vertices3 = []

for j in range(19):
    rnd = random.randint(0,1)
    if(rnd == 0):
        h = hexagon
    else:
        h = hexagon2
            
    for ve in h:    
        vertex = [ve[0] + v[j,0], ve[1] + v[j,1]]
        vertices3.append(vertex)


v2 = np.array(vertices2)
v3 = np.array(vertices3)


px = v[:,0]
px = np.append(px,v2[:,0]) 
px = np.append(px,v3[:,0])

py = v[:,1]
py = np.append(py,v2[:,1]) 
py = np.append(py,v3[:,1])


pos = np.zeros((373,2))
pos[:,0] = px
pos[:,1] = py

pos = pos.round(5)
pos = np.unique(pos, axis=0)

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.axes().autoscale_view()
plt.plot(pos[:,0], pos[:,1], "o")




#%%
#### GET A SQUARE FRAME OF THIS 
cutoff = 5.0981

control = []
for i in range(pos.shape[0]):
    p = pos[i,:]
    if (abs(p[0]) < cutoff and abs(p[1]) < cutoff):
        control.append(p)
control = np.array(control)

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.axes().autoscale_view()
plt.xlim(-cutoff,cutoff)
plt.ylim(-cutoff,cutoff)
plt.plot(control[:,0], control[:,1], "o")

            


#%%
####### SECOND INFLATION

v = lamb * control


vertices2 = []

theta = np.radians(15.0)
s=np.sin(theta)
t=np.tan(theta)

vertex = [0.5/t,0.5]
vertices2.append(vertex)
for i in range(11):
    vertex = rotate_dodeca(vertex)
    vertices2.append(vertex)

temp = list(vertices2)

for j in range(v.shape[0]):
    for ve in temp:
        vertex = [ve[0] + v[j,0], ve[1] + v[j,1]]
        vertices2.append(vertex)

v2 = np.array(vertices2)


vertices3 = []

for j in range(v.shape[0]):
    rnd = random.randint(0,1)
    if(rnd == 0):
        h = hexagon
    else:
        h = hexagon2
            
    for ve in h:    
        vertex = [ve[0] + v[j,0], ve[1] + v[j,1]]
        vertices3.append(vertex)

v3 = np.array(vertices3)


px = v[:,0]
px = np.append(px,v2[:,0]) 
px = np.append(px,v3[:,0])

py = v[:,1]
py = np.append(py,v2[:,1]) 
py = np.append(py,v3[:,1])

pos = np.zeros((len(px),2))
pos[:,0] = px
pos[:,1] = py

pos = pos.round(5)
pos = np.unique(pos, axis=0)


plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.axes().autoscale_view()
plt.plot(pos[:,0], pos[:,1], "ro")


#%%
#### GET A SQUARE FRAME OF THIS 
cutoff = 7.46412

control = []
for i in range(pos.shape[0]):
    p = pos[i,:]
    if (abs(p[0]) <= cutoff and abs(p[1]) <= cutoff):
        control.append(p)
control = np.array(control)

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.axes().autoscale_view()
plt.xlim(-cutoff,cutoff)
plt.ylim(-cutoff,cutoff)
plt.plot(control[:,0], control[:,1], "o")



#%%%
tdens = np.linspace(0.6,1.5,100)
pars = np.zeros((100,3))

n = len(control)

for d in range(len(tdens)):
    
    x = np.sqrt(n / tdens[d]) / (2. * cutoff)
    L = (2. * cutoff) * x
    control2 = x * control + L/2
    
    ec = []
    ec.append(control2[0,:])
    
    for i in range(1,len(control2)):
        
        if ((control2[i,:] >= L).sum() == 0):
            
            p1 = control2[i,:]
            flag = [False,False,False,False]
        
            for j in range(len(ec)):
                
                p2 = np.array(ec[j])
                
                r = p1 - p2
                dr = np.sum(r*r)
                if(dr <= 1.0):
                    flag[0] = True
                
                sx = np.array([p1[0]-L,p1[1]])
                r = p2 - sx
                dr = np.sum(r*r)
                if(dr <= 1.0):
                    flag[1] = True
            
                sy = np.array([p1[0],p1[1]-L])
                r = p2 - sy
                dr = np.sum(r*r)        
                if(dr <= 1.0):
                    flag[2] = True
                    
                sd = np.array([p1[0]-L,p1[1]-L])
                r = p2 - sd
                dr = np.sum(r*r)        
                if(dr <= 1.0):
                    flag[3] = True
                
            if(not any(flag)):
                ec.append(p1)
                
    ec = np.array(ec)
    ec = np.unique(ec, axis=0)

                
    for i in range(len(ec)):
        
            p1 = ec[i,:]
            flag = [False,False,False,False]
        
            for j in range(len(ec)):
                
                p2 = ec[j,:]
                
                r = p1 - p2
                dr = np.sum(r*r)
                if(dr <= 1.0 and i!=j):
                    flag[0] = True
                
                sx = np.array([p1[0]-L,p1[1]])
                r = p2 - sx
                dr = np.sum(r*r)
                if(dr <= 1.0 and i!=j):
                    flag[1] = True
            
                sy = np.array([p1[0],p1[1]-L])
                r = p2 - sy
                dr = np.sum(r*r)        
                if(dr <= 1.0 and i!=j):
                    flag[2] = True
                    
                sd = np.array([p1[0]-L,p1[1]-L])
                r = p2 - sd
                dr = np.sum(r*r)        
                if(dr <= 1.0 and i!=j):
                    flag[3] = True
                
            if(any(flag)):
                print(p1)
                print(p2)
    
    
    density = len(ec) / (L*L)
    pars[d,0] = density
    pars[d,1] = len(ec)
    pars[d,2] = L
    
    positions = pd.DataFrame(ec)
    print("Density: ", density, "   L = ", L, "   N = ", len(ec) )
    pars
    
    filename = "qclatt_" + str(d) + ".inp"
    positions.to_csv(filename,header=False,index=False,sep='\t', float_format='%.6E')


pars = pd.DataFrame(pars)
pars.to_csv("qclatt.pars",header=False,index=True,sep='\t', float_format='%.4E')

#%%
def plot(**kwargs):
    plt.figure()
    c='g'
    c2='r'
    vmin=None
    vmax=None
    
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    x_cm = np.array(positions.iloc[:,0])
    y_cm = np.array(positions.iloc[:,1])
    hc = 0.5
    sc = 0.0
    
    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    
    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x_cm, y_cm, hc)]
    collection = PatchCollection(patches, facecolors='g', edgecolors='black')
        
    patches2 = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x_cm, y_cm, sc)]
    collection2 = PatchCollection(patches2, facecolors=(1,0,0,0.4))
    
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)
        collection2.set_array(np.asarray(c2))
        collection2.set_clim(vmin, vmax)
    
    ax = plt.gca()
    plt.ylim(0,L)
    plt.xlim(0,L)
    
    plt.title(r"Archimedean tiling, N=%d, $\rho = %.2f$" % (len(ec),density), fontsize=16)
    plt.xticks([],[])
    plt.yticks([],[])
    
    plt.gca().set_aspect('equal', adjustable='box')
    ax.add_collection(collection2)
    ax.add_collection(collection)
    ax.autoscale_view()
    filename = "centroid_trlatt.pdf"
    plt.savefig(filename,bbox_inches = "tight")
    
    if c is not None:
        plt.sci(collection)
#%%
plot()



