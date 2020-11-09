from __future__ import division
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.stats as stats
import warnings
from sklearn.mixture import GaussianMixture

np.seterr(divide='ignore', invalid='ignore')

n = 2  # number of feature dimensions
N = 10000  # number of samples
l = 2  # number of labels

# class-conditional pdf parameters for L = 0:
m0 = np.array([11, 0])
C0 = np.array([[4, 0], [0, 2]])

# class-conditional pdf parameters for L = 1
m1 = np.array([1, 2])
C1 = np.array([[2, 0], [0, 2]])

# class priors for L = 0 and L = 1 respectively
p = np.array([0.6, 0.4])


# make data for k means implementation
def make_data():

    # create N samples of data
    label_probabilities = np.random.rand(N)
    label = np.zeros(N, dtype=np.int32)

    N0 = 0
    N1 = 0

    ind_1 = []
    ind_0 = []

    for i in range(N):
        label[i] = 1 if label_probabilities[i] <= p[0] else 0
        if label[i] == 1:
            N1 = N1 + 1
            ind_1.append(i)

        if label[i] == 0:
            N0 = N0 + 1
            ind_0.append(i)

    x = np.zeros((N, n))
    x0 = np.random.multivariate_normal(m0, C0, N0)
    x1 = np.random.multivariate_normal(m1, C1, N1)

    for i in range(0, N0 - 2, 2):
        x[ind_0[i]] = x0[i]

    for i in range(N1):
        x[ind_1[i]] = x1[i]

    N0_file = "N0_1000.txt"
    N1_file = "N1_1000.txt"

    f1 = open(N1_file, "a")
    f0 = open(N0_file, "a")

    # erase contents of files to reset them
    f1 = open(N1_file, 'r+')
    f1.truncate(0)
    f0 = open(N0_file, 'r+')
    f0.truncate(0)

    f1.write(str(N1))
    f0.write(str(N0))

    f1.close()
    f0.close()

    discriminant_score = np.zeros(N, dtype=np.float)
    # define the discriminant score for each data point
    for i in range(N):
        score = (math.log(stats.multivariate_normal.pdf(x[i], m0, C0)) - math.log(stats.multivariate_normal.pdf(x[i], m1, C1)))
        discriminant_score[i] = score

    """
    Plot data
    """
    fig = plt.figure(figsize=(10, 7))
    plt.xlabel('parameter 1')
    plt.ylabel('parameter 2')
    plt.scatter(x0[:, 0], x0[:, 1], marker='x', color='green', linewidths=1)
    plt.scatter(x1[:, 0], x1[:, 1], marker='.', color='blue', linewidths=1)
    plt.title('Q1 Validation Data')

    plt.figure(figsize=(10, 7))
    plt.plot(np.linspace(0, N, N), discriminant_score)


    """
    Save data to corresponding text files
    """
    np.savetxt("desc_score_1000.csv", discriminant_score, delimiter=",")
    np.savetxt("label_1000.csv", label, delimiter=",")
    np.savetxt("x_1000.txt", x, delimiter=",")


def calculate_distance(k, point, means):

    dist = []
    dist_row = []
    for i in range(k):
        dist.append(np.linalg.norm(point - means[i]))


    #print "x: "
    #print point
    #print "dist: "
    #print dist
    return dist


def k_means(k):

    f0 = open("N0_1000.txt", "r")
    f1 = open("N1_1000.txt", "r")
    N0 = float(f0.read())
    N1 = float(f1.read())
    N = int(N0 + N1)

    discriminant_score = np.genfromtxt('desc_score_1000.csv', delimiter=",")
    label = np.genfromtxt('label_1000.csv', delimiter='\n')
    x = np.genfromtxt('x_1000.txt', delimiter=',')

    # set random initial means of clusters as random points from each cluster
    est_means = []

    # matrix of estimated labels for each point
    k_labels = np.zeros(N)

    for i in range(k):
        rand_mean_ind = random.randint(0, N)
        est_means.append(x[rand_mean_ind])

    print "estimated means before: "
    print est_means

    converged = 0
    while converged != N:
        # find distances
        # matrix of distances of each point to each mean of k clusters
        distance = []
        for point in x:
            distance.append(calculate_distance(k, point, est_means))

        converged = 0
        for i in range(N):
            # min distance for each row
            min_dist_ind = np.argmin(distance[i])
            # set label to cluster with min distance
            if k_labels[i] == min_dist_ind:
                converged = converged + 1

            k_labels[i] = min_dist_ind

        # update means for each cluster
        for i in range(k):
            sample = []
            for j in range(N):
                if k_labels[j] == i:
                    sample.append(x[j])

            est_means[i] = np.mean(sample, axis=0)

    # plot data
    # lists of each label
    miss = []
    k_labels_1 = []
    k_labels_0 = []

    for i in range(N):
        if k_labels[i] != label[i]:
            miss.append(x[i])

        elif k_labels[i] == 1:
            k_labels_1.append(x[i])

        elif k_labels[i] == 0:
            k_labels_0.append(x[i])

    miss = np.asarray(miss)
    k_labels_0 = np.asarray(k_labels_0)
    k_labels_1 = np.asarray(k_labels_1)

    print "means after: "
    print est_means

    plt.figure(figsize=(10, 7))
    plt.scatter(miss[:, 0], miss[:, 1], color='red', marker='v', linewidths=1)
    plt.scatter(k_labels_0[:, 0], k_labels_0[:, 1], color='blue', marker='.', linewidths=1)
    plt.scatter(k_labels_1[:, 0], k_labels_1[:, 1], color='green', marker='x', linewidths=1)
    plt.show()


make_data()
k_means(2)
