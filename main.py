#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:56:47 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;

import scipy.io as scio;
import matplotlib.image as img;
import matplotlib.pyplot as plt;
import matplotlib.gridspec as gs;

class unif_metro:
    
    def __init__(self,f,df,beta,l,h):
        self.f  =  f;
        self.df = df;
        
        self.b = beta;
        
        self.l = l;
        self.h = h;
        
        self.Z = 1e-30;
        self.N = 1e-3;
        for n in range(100):
            self.N += 1e-3;
            a = (self.h-self.l)*rd.rand(self.l.size)+self.l;
            self.Z += np.exp(-beta*f(a));
        self.Z /= self.N;
        
    def get_p(self,x):
        a = np.exp(-self.b*self.f(x));
        self.Z = (self.N*self.Z+a)/(self.N+1e-3);
        self.N += 1e-3;
        return (a/self.N)/self.Z;
    def trail(self,T,tau):
        
        x = np.zeros((T,self.l.size));        
        x[0] = (self.h-self.l)*rd.rand(self.l.size)+self.l;
        
        it = 0;
        while it < T-1:
            
            px = self.get_p(x[it]);
            
            dx = -self.b*(1.0-px)*self.df(x[it]);

            y = x[it]+0.5*tau**2*dx+tau*rd.randn(self.l.size);
            
            py = self.get_p(y);
            dy = -self.b*(1.0-py)*self.df(y);
            
            print it;            
            q_yx = np.exp(-0.5*np.linalg.norm(y-x[it]-0.5*(tau**2)*dx)**2/(tau**2));            
            q_xy = np.exp(-0.5*np.linalg.norm(x[it]-y-0.5*(tau**2)*dy)**2/(tau**2));
            print px,py;
            print q_xy,q_yx;

            a = min(1,(py*q_xy)/(px*q_yx));
            print a;
            if rd.rand(1) <= a:
                it += 1;
                
                x[it] = y;
        return x;
    def samp(self,T,tau):
        
        x = (self.h-self.l)*rd.rand(self.l.size)+self.l;
        
        it = 0;
        while it < T-1:
            
            px = self.get_p(x);
            
            dx = -self.b*(1.0-px)*self.df(x);

            y = x+0.5*tau**2*dx+tau*rd.randn(self.l.size);
            
            py = self.get_p(y);
            dy = -self.b*(1.0-py)*self.df(y);
            
            #print it;            
            q_yx = np.exp(-0.5*np.linalg.norm(y-x-0.5*(tau**2)*dx)**2/(tau**2));            
            q_xy = np.exp(-0.5*np.linalg.norm(x-y-0.5*(tau**2)*dy)**2/(tau**2));
            #print px,py;
            #print q_xy,q_yx;

            a = min(1,(py*q_xy)/(px*q_yx));
            #print a;
            if rd.rand(1) <= a:
                x = y;
            it += 1;
                
        return x;

def dist_s1(a):
    return np.minimum(2.0*np.pi-np.abs(a),np.abs(a));

def ddist_s1(a):
    if np.abs(a) < np.pi:
        return np.sign(a);
    else:
        return -np.sign(a);

def gau(x,y,p,t,sp,st):
    dp = (np.log(x**2+y**2)-p)/sp;   
    t_ = np.arctan2(y,x);
    t_ = 2*np.pi+t_ if t_ < 0 else t_; 
    dt = dist_s1(t_-t)/st;
    return np.exp(-0.5*(dp**2+dt**2));

def dgau(x,y,p,t,sp,st):
    dp = (np.log(x**2+y**2)-p)/sp;

    t_ = np.arctan2(y,x);
    t_ = 2.0*np.pi+t_ if t_ < 0 else t_;
    dt = dist_s1(t_-t)/st;
    
    g_ = np.exp(-0.5*(dp**2+dt**2));
    
    dgp = -g_*dp/sp;
    dgt = -g_*dt*ddist_s1(t_-t)/st;
    
    dpx = 2.0*x/(x**2+y**2);
    dpy = 2.0*y/(x**2+y**2);
    dtx = -y/(x**2+y**2);
    dty = x/(x**2+y**2);
    return np.array([dgp*dpx + dgt*dtx,dgp*dpy + dgt*dty]);
    
def f(x):
    return 0.25*gau(x[0],x[1],0.0,0.25*np.pi,1.5,0.5)+0.35*gau(x[0],x[1],1.0,1.35*np.pi,0.3,0.75)+0.4*gau(x[0],x[1],2.0*np.log(2.0),0.55*np.pi,0.5,0.25);
def df(x):
    return 0.25*dgau(x[0],x[1],0.0,0.25*np.pi,1.5,0.5)+0.35*dgau(x[0],x[1],1.0,1.35*np.pi,0.3,0.75)+0.4*dgau(x[0],x[1],2.0*np.log(2.0),0.55*np.pi,0.5,0.25);

L = 128;
l = np.linspace(-5.0,5.0,L);
A = np.zeros((L,L,3));
for i in range(L):
    for j in range(L):
        A[i,j,0 ] = f(np.array( [l[j],l[i]] ));
        A[i,j,1:] = df(np.array([l[j],l[i]] ));

fig = plt.figure();
plt.imshow(A[:,:,0],cmap='terrain'),plt.quiver(np.arange(L),np.arange(L),A[:,:,1], A[:,:,2],pivot='mid');
plt.show();

metro = unif_metro(lambda x: -np.log(f(x)),lambda x: -df(x)/f(x),1.0,-5.0*np.ones(2),5.0*np.ones(2));

N = 1000;
X = metro.trail(N,1e-3);

fig = plt.figure();
plt.quiver(l,l,A[:,:,1],A[:,:,2]),plt.scatter(X[:,0],X[:,1],s=5,c=np.arange(N),cmap='jet');
plt.show();

N = 1000;
Y = np.zeros((N,2));
F = np.zeros(N);
for n in range(N):
    print n;
    Y[n] = metro.samp(100,1e-1);
    F[n] = f(Y[n]);

L = 128;
l = np.linspace(np.min(Y),np.max(Y),L);
A = np.zeros((L,L,3));
for i in range(L):
    for j in range(L):
        A[i,j,0 ] = f(np.array( [l[j],l[i]] ));
        A[i,j,1:] = df(np.array([l[j],l[i]] ));

fig = plt.figure();
plt.imshow(A[:,:,0],cmap='gray'),plt.scatter((L-1)*(Y[:,0]-np.min(l))/(np.max(l)-np.min(l)),(L-1)*(Y[:,1]-np.min(l))/(np.max(l)-np.min(l)),s=5,c=F,cmap='jet');
plt.show();