#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2015/04/13

@author: drumichiro
'''
import numpy as np
import pylab as plt
import matplotlib.mlab as mlab
from myutil.sampler import *


def plot1dSample(sample):
    bins = np.ceil(max(sample) - min(sample))*4
    plt.hist(sample, bins, fc="b")


def plot2dSample(sample):
    x, y = sample.T
    plt.plot(x, y, "bo")


def plot1dGaussian(mu, sigma, sample, ratio):
    xMin = np.floor(np.amin(sample))
    xMax = np.ceil(np.amax(sample))
    samples = len(sample)
    grid = np.linspace(xMin, xMax, samples)

    weight = float(samples)/np.sqrt(2*np.pi)
    dist = np.zeros(samples)
    for i1 in np.arange(0, len(mu)):
        dist += gaussian1d(grid, mu[i1], sigma[i1])*weight*ratio[i1]
    plt.plot(grid, dist, "g", linewidth=2)


def plot2dGaussian(mu, sigma, sample, ratio):
    delta = 0.025  # Sampling rate of contour.
    xMin, yMin = np.floor(np.amin(sample, axis=0))
    xMax, yMax = np.ceil(np.amax(sample, axis=0))
    X, Y = np.meshgrid(np.arange(xMin, xMax, delta),
                       np.arange(yMin, yMax, delta))
    grid = np.array([X.T, Y.T]).T

    circleRate = 3  # Adjust size of a drawn circle.
    Z = 0
    for i1 in np.arange(0, len(mu)):
        Z += gaussian2d(grid, mu[i1], sigma[i1]*circleRate)*ratio[i1]
    plt.contour(X, Y, Z, 2, linewidths=2, colors="g")


def plot1dSampleAndGaussian(sample, mu, sigma, ratio):
    plot1dSample(sample)
    plot1dGaussian(mu, sigma, sample, ratio)
    plt.show()


def plot2dSampleAndGaussian(sample, mu, sigma, ratio):
    plot2dSample(sample)
    plot2dGaussian(mu, sigma, sample, ratio)
    plt.show()
