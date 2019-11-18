#! /usr/bin/env python
import os
import numpy as np
from numpy import maximum,minimum,NaN,Inf,arange,isscalar,array
from pyDFTutils.math.peakdetect import peakdetect
import sys
from numpy import *
from functools import reduce
def get_potential():
    if not os.path.exists('./vplanar.txt'):
        raise IOError('No data vplanar.txt found. Please run work_function')
    data=np.loadtxt('vplanar.txt',skiprows=1)
    pos=data[:,0]
    pot=data[:,1]
    return pos,pot


def periodic_average(data,step):
    l=len(data)
    avg=data.copy()
    data=np.reshape(data,[1,l])
    tri_data=np.repeat(data,3,axis=0).flatten()
    for i in range(l):
        print(i)
        l1=-step/2+i+l
        l2=step/2+i+l
        avg[i]=np.average(tri_data[l1:l2])
    return avg


def periodic_average_dynamic(data):
    p=array(peakdetect(data,lookahead=5,delta=0.01))
    N=len(data)
    xmin=p[0][:,0][::1]
    steps=[N-xmin[-1]+xmin[0]]
    x_range=[(xmin[-1]-N,xmin[0])]

    for ix in range(len(xmin)-1):
        x_range.append((xmin[ix],xmin[ix+1]))
        steps.append(xmin[ix+1]-xmin[ix])
    x_range.append((xmin[-1],xmin[0]+N))
    steps.append(xmin[0]+N-xmin[-1])

    avg=data.copy()
    data=np.reshape(data,[1,N])
    tri_data=np.repeat(data,3,axis=0).flatten()
    for i in range(N):
        for xr,step in zip(x_range,steps):
            if xr[0]<=i<xr[1]:
                #l1=-step/2+i+N
                #l2=step/2+i+N-1
                l1=xr[0]+N
                l2=xr[1]+1+N
        avg[i]=np.average(tri_data[l1:l2])
    return avg






def peaks(data, step):
    n = len(data) - len(data)%step # ignore tail
    slices = [ data[i:n:step] for i in range(step) ]
    peak_max = reduce(maximum, slices)
    peak_min = reduce(minimum, slices)
    return np.transpose(np.array([peak_max, peak_min]))


def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    Returns two arrays
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    % [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    % maxima and minima ("peaks") in the vector V.
    % MAXTAB and MINTAB consists of two columns. Column 1
    % contains indices in V, and column 2 the found values.
    %
    % With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    % in MAXTAB and MINTAB are replaced with the corresponding
    % X-values.
    %
    % A point is considered a maximum peak if it has the maximal
    % value, and was preceded (to the left) by a value lower by
    % DELTA.
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    """
    maxtab = []
    mintab = []
    if x is None:
        x = arange(len(v))
    v = array(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    lookformax = True
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mxpos-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

def test(step=None, ):
    pos,pot=get_potential()
    p=array(peakdetect(pot,lookahead=5,delta=0.01))
    zero=p[0][0,1]
    pot=pot-zero

    N=len(pot)
    xmin=p[1][:,0]
    step=xmin[2]-xmin[0]-1

    import matplotlib.pyplot as plt
    plt.plot(pot,color='blue')
    plt.plot(periodic_average(pot,step),color='red')
    plt.plot(periodic_average_dynamic(pot),color='purple')
    plt.scatter(p[0][:,0],p[0][:,1]-zero,color='red')
    plt.scatter(p[1][:,0],p[1][:,1]-zero,color='green')
    plt.xlim(0,N)
    #plt.xticks(p[1][:,0][::2],xl)
    plt.grid()
    plt.show()
    pos_S=p[1][:,0][1::2]
    spos=list(pos_S[1:])
    spos.append(240)
    spos=np.array(spos)
    print((spos-pos_S)/480*38.1814)
if __name__=='__main__':
    test()
