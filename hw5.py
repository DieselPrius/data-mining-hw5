from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab
import graphviz
import pickle
from sklearn.neighbors import KNeighborsClassifier
import random
import math


class centroid:
    def __init__(self,x,y):
        self.m_x = x
        self.m_y = y


class Team:
    m_clabel = ""
    def __init__(self,name,x,y):
        self.m_name = name
        self.m_x = x
        self.m_y = y

    def print(self):
        print(str(self.m_name) + " " + str(self.m_x) + " " + str(self.m_y) + " " + self.m_clabel)


def manhattan(team,centroid):
    return (abs(team.m_x - centroid.m_x) + abs(team.m_y - centroid.m_y))

def euclidean(team,centroid):
    return math.sqrt( ((team.m_x - centroid.m_x)*(team.m_x - centroid.m_x)) + ((team.m_y - centroid.m_y)*(team.m_y - centroid.m_y)) )

def calcCentroid(teamCluster):
    n = len(teamCluster)
    xsum = 0
    ysum = 0
    for team in teamCluster:
        xsum += team.m_x
        ysum += team.m_y
    return centroid(xsum/n,ysum/n)


def kmeansManhattan(c1x,c1y,c2x,c2y,teams):
    prev_c1 = centroid(c1x,c1y)
    prev_c2 = centroid(c2x,c2y)
    new_c1 = centroid(0,0)
    new_c2 = centroid(0,0)

    iterationCount = 1
    while (round(prev_c1.m_x,5) != round(new_c1.m_x,5) and round(prev_c1.m_y,5) != round(new_c1.m_y,5)) or (round(prev_c2.m_x,5) != round(new_c2.m_x,5) and round(prev_c2.m_y,5) != round(new_c2.m_y,5)):
        if(iterationCount != 1):
            prev_c1 = new_c1
            prev_c2 = new_c2
        #assign points to their closest centroids
        for team in teams:
            distToC1 = manhattan(team,prev_c1)
            distToC2 = manhattan(team,prev_c2)
            if(distToC1 <= distToC2):
                team.m_clabel = "1"
            else:
                team.m_clabel = "2" 
        #recompute centroid for each cluster
        cluster1 = []
        cluster2 = []
        for team in teams:
            if(team.m_clabel == "1"):
                cluster1.append(team)
            else:
                cluster2.append(team)
        new_c1 = calcCentroid(cluster1)
        new_c2 = calcCentroid(cluster2)
        iterationCount += 1

    return (cluster1, cluster2)


def kmeansEuclidean(c1x,c1y,c2x,c2y,teams):
    prev_c1 = centroid(c1x,c1y)
    prev_c2 = centroid(c2x,c2y)
    new_c1 = centroid(0,0)
    new_c2 = centroid(0,0)

    iterationCount = 1
    while (round(prev_c1.m_x,5) != round(new_c1.m_x,5) and round(prev_c1.m_y,5) != round(new_c1.m_y,5)) or (round(prev_c2.m_x,5) != round(new_c2.m_x,5) and round(prev_c2.m_y,5) != round(new_c2.m_y,5)):
        if(iterationCount != 1):
            prev_c1 = new_c1
            prev_c2 = new_c2
        #assign points to their closest centroids
        for team in teams:
            distToC1 = euclidean(team,prev_c1)
            distToC2 = euclidean(team,prev_c2)
            if(distToC1 <= distToC2):
                team.m_clabel = "1"
            else:
                team.m_clabel = "2" 
        #recompute centroid for each cluster
        cluster1 = []
        cluster2 = []
        for team in teams:
            if(team.m_clabel == "1"):
                cluster1.append(team)
            else:
                cluster2.append(team)
        new_c1 = calcCentroid(cluster1)
        new_c2 = calcCentroid(cluster2)
        iterationCount += 1

    return (cluster1, cluster2)


def printResults(cluster1,cluster2):
    print("     cluster 1 = ",end="")
    for team in cluster1:
        print(team.m_name,end=" ")
    print("")
    print("     cluster 2 = ",end="")
    for team in cluster2:
        print(team.m_name,end=" ")
    print("\n")



#pre-processing
data = pd.read_csv("data.csv")
teamNames = data["Team"].values
xvals = data["x"].values
yvals = data["y"].values
teams = []
for name in teamNames:
    teams.append(Team(name,0,0))
i = 0
for x in xvals:
    teams[i].m_x = x
    i += 1
i = 0
for y in yvals:
    teams[i].m_y = y
    i += 1


print("Problem #1")

print("part 1 results:")
(cluster1, cluster2) = kmeansManhattan(4,6,5,4,teams)
printResults(cluster1,cluster2)


print("part 2 results:")
(cluster1, cluster2) = kmeansEuclidean(4,6,5,4,teams)
printResults(cluster1,cluster2)

print("part 3 results:")
(cluster1, cluster2) = kmeansManhattan(3,3,8,3,teams)
printResults(cluster1,cluster2)

print("part 4 results:")
(cluster1, cluster2) = kmeansManhattan(3,2,4,8,teams)
printResults(cluster1,cluster2)