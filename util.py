#! -*- coding:utf-8 -*-

import os
import sys
import cv2
import numpy as np
import pylab

def _resizing(img):
    #return cv2.resize(img, (256, 256))
    return cv2.resize(img, (32, 32))

def _reg(img):
    return img/127.5 - 1.0

def _re_reg(img):
    return (img + 1.0) * 127.5

def get_figs(target_dir):
    ret = []
    for file_name in os.listdir(target_dir):
        target_file = os.path.join(target_dir, file_name)
        img = cv2.imread(target_file, 0)
        ret.append(_reg(_resizing(img)))
    return np.asarray(ret, dtype = np.float32)

def dump_figs(figs, dump_dir):
    for i, fig in enumerate(figs):
        target_file = os.path.join(dump_dir, '{}.jpg'.format(i))
        cv2.imwrite(target_file, _re_reg(fig))

def plot_scatter(data, dir=None,  filename="scatter", disc_data = None, color="blue"):
    
    #if dir is None:
    #    raise Exception()
    try:
        os.mkdir(dir)
    except:
        pass
    fig = pylab.gcf()
    fig.set_size_inches(16.0, 16.0)
    pylab.clf()
    pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color=color)
    
    if disc_data is not None:
        pylab.scatter(disc_data[:, 0], disc_data[:, 1], s=20, marker="x", edgecolors="none", color='red')        
    pylab.xlim(-6, 6)
    pylab.ylim(-6, 6)
    pylab.savefig("{}/{}.png".format(dir, filename))        

def plot_scatter2(data_x, data_y, dir=None,  filename="scatter", color="blue"):

    try:
        os.mkdir(dir)
    except:
        pass
    fig = pylab.gcf()
    fig.set_size_inches(16.0, 16.0)
    pylab.clf()
    #pylab.scatter(data_x, data_y, s=40, marker="o", edgecolors="none", color=color)
    #pylab.scatter(data_x, data_y, color=color, linewidths = 3)
    pylab.plot(data_x, data_y, color=color, linewidth = 5)
    
    pylab.xlim(0.0, 5.0)
    pylab.ylim(-0.05, 0.05)
    pylab.savefig("{}/{}.png".format(dir, filename))

def plot_scatter3(data_x, data_y, dir=None,  filename="scatter", color="blue"):

    try:
        os.mkdir(dir)
    except:
        pass
    fig = pylab.gcf()
    fig.set_size_inches(16.0, 16.0)
    pylab.clf()
    #pylab.scatter(data_x, data_y, s=40, marker="o", edgecolors="none", color=color)
    pylab.plot(data_x, data_y, color=color, linewidth = 5)

    pylab.xlim(0.0, 5.0)
    pylab.ylim(-0.8, -0.6)
    pylab.savefig("{}/{}.png".format(dir, filename))                
