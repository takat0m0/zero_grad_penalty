# -*- coding:utf-8 -*-

import os
import sys
import numpy as np

import pylab

if __name__ == '__main__':
    file_name = 'zero2.txt'
    nums = []
    ret = []
    num = 0

    with open(file_name, 'r') as f:
        for l in f:
            l = l.strip()
            if l.startswith('**'):
                continue
            else:
                tmp = l.split(' ')[0]
                nums.append(float(num))
                ret.append(float(tmp))
                num += 1
    ret = np.asarray(ret)
    nums = np.asarray(nums)
                
    fig = pylab.gcf()
    fig.set_size_inches(16.0, 9.0)
    pylab.clf()                
    
    pylab.plot(nums, ret, color='blue', linewidth = 5)
    
    pylab.xlim(0.0, 240.0)
    #pylab.ylim(-0.1, 0.1)
    pylab.ylim(-0.5, 5.0)
    pylab.savefig('hoge.png')
