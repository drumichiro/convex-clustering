#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2015/04/13

@author: drumichiro
'''
import numpy as np
from myutil.sampler import *
from show_clustering_image import *


def calculateLikelihoodCombination(x, sigma, gaussianFunc):
    likelihood = np.empty(0)
    for i1 in x:
        likelihood = np.append(likelihood, gaussianFunc(x, i1, sigma))
    samples = len(x)
    return np.reshape(likelihood, (samples, samples))


def calculatePrior(prePrior, likelihood):
    samples = len(prePrior)
    molecule = (prePrior*likelihood.T).T
    denomin = np.sum(molecule, axis=0) + 1e-10
    return np.sum(molecule/denomin, axis=1)/samples


def rejectWeakCentroid(x, prior, likelihood):
    samples = len(x)
    # I decided the threshold based on a paper
    #  about convex clustering.
    threshold = 0.001/samples
    strong = prior > threshold
    resamples = np.count_nonzero(strong)
    strongMat = np.array(np.matrix(strong).T*np.matrix(strong))
    return x[strong], prior[strong], \
        likelihood[strongMat].reshape(resamples, resamples)


def selectValidCentroid(x, prior, threshold):
    samples = len(x)
    mu = np.empty(0)
    ratio = np.empty(0)
    for rows in range(samples):
        index = np.argmax(prior)
        if prior[index] < threshold:
            if 0 < rows:
                cols = mu.size/rows
                mu = np.reshape(mu, (rows, cols))
            break
        mu = np.append(mu, x[index])
        ratio = np.append(ratio, prior[index])
        x = np.delete(x, index, axis=0)
        prior = np.delete(prior, index)
    return mu, ratio


def runConvexClustering(x, hyperSigma, gaussian, plotter):
    # Pre-calculation of likelihood
    likelihood = calculateLikelihoodCombination(x, hyperSigma, gaussian)
    # Initialize prior probability.
    classNum = len(x)  # Number of samples is used as number of classes.
    firstPrior = 1.0/classNum  # Set non zero value.
    prior = np.repeat(firstPrior, classNum)
    xForPlotter = x

    # Update parameters.
    for dummy in range(1000):
        prior = calculatePrior(prior, likelihood)
        # This procedure is optimization.
        # I have little confidence of using the below function...
        # x, prior, likelihood = rejectWeakCentroid(x, prior, likelihood)

    # Half an initial prior is used as threshold.
    threshold = 0.01
    estMu, ratio = selectValidCentroid(x, prior, threshold)
    estClassNum = len(estMu)
    maxSigmaForPrinting = np.max(hyperSigma)
    print "The number of used centroids is (%d/%d) using sigma:(%.2f)." \
        % (estClassNum, classNum, maxSigmaForPrinting)
    constSigma = np.array([hyperSigma, ]*estClassNum)
    plotter(xForPlotter, estMu, constSigma, ratio)


def testUsingSharedCase(x, geneSigma, gaussian, plotter):
    # The same(small) value as the variance used in sampling
    #  leads to not good estimation.
    hyperSigma = geneSigma
    runConvexClustering(x, hyperSigma, gaussian, plotter)

    # Slightly larger hyper parameter than the variance used in sampling.
    # It is for good estimation.
    hyperSigma = geneSigma*2
    runConvexClustering(x, hyperSigma, gaussian, plotter)

    # Too large hyper parameter also leads to bad estimation.
    hyperSigma = geneSigma*10
    runConvexClustering(x, hyperSigma, gaussian, plotter)


def testConvexClustering1d():
    # Average and variance of distribution of sample data.
    geneMu = np.array([10, 20, 35, 45])
    geneSigma = np.array([1, 1, 1, 1])

    # Total # of samples is <len(geneMu) * samples>
    samplesPerDist = 100
    x = generate1dSample(samplesPerDist, geneMu, geneSigma)

    testUsingSharedCase(x, geneSigma[0], gaussian1d, plot1dSampleAndGaussian)


def testConvexClustering2d():
    # Average and variance of distribution of sample data.
    geneMu = np.array([[10, 20],
                       [20, 10],
                       [30, 20],
                       [20, 30],
                       ])
    geneSigma = np.array([[[2, 0],
                           [0, 2]],
                          [[2, 0],
                           [0, 2]],
                          [[2, 0],
                           [0, 2]],
                          [[2, 0],
                           [0, 2]]
                          ])

    # Total # of samples is <len(geneMu) * samples>
    samplesPerDist = 100
    x = generate2dSample(samplesPerDist, geneMu, geneSigma)

    testUsingSharedCase(x, geneSigma[0], gaussian2d, plot2dSampleAndGaussian)


if __name__ == "__main__":
    testConvexClustering1d()
    testConvexClustering2d()
    print "Done."
