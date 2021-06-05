# -*- coding: utf-8 -*-
#python clustering.py data-3/input1.txt 8 15 22
#python clustering.py input1.txt 8 15 22
import sys
import itertools
import numpy as np
import pandas as pd
import time
from queue import Queue #put, get
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, data, n, eps, minpts, input_f):
        self.input_string = input_f
        self.N = n 
        self.Eps = eps
        self.Minpts = minpts
        self.D = data
        self.Data = {}
        self.Data_size = data.shape[0]

        # dataframe is tooooooo slow to get neighbor
        # so i use dictionary (hash)
        for i in range(self.Data_size):
            d = data.iloc[i]
            self.Data[d['key']] = (d['x'], d['y'])

        self.Neighbors = {}
        self.Neighbors = {int(key): [] for key,xy_tuple in self.Data.items()}

        self.get_neighbor()
        # find directly density reachable relationships
        # it's like vector in cpp

        self.Core = [key for key, neighbors in self.Neighbors.items() if len(neighbors) >= self.Minpts]
        # list comprehension is used to pick up core point

        # first point is check the point if it is contained in any group
        # i can use dfs or bfs or union find to grouping
        # list? dictionary?
        # second point is how to represent the group
        # is it possible that using set to union every point?

        self.Clusters = []
        self.cluster()

        # for c in self.Clusters: print(len(c))
        self.label = [-1 for i in range(self.Data_size)]
        self.write_clusters()

        self.D['label'] = self.label
        groups = self.D.groupby('label')

        fig, ax = plt.subplots()
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            ax.plot(group['x'], group['y'], marker='.', linestyle='', ms=12, label=name)
        ax.legend()

        plt.show()

    def write_clusters(self):
        for i in range(len(self.Clusters)):
            file_name = self.input_string + "_cluster_" + str(i) + ".txt"
            f = open(file_name, 'w')
            self.Clusters[i].sort()
            for point in self.Clusters[i]:
                self.label[point] = i
                f.write(str(point)+'\n')
            f.close()

    def get_distance(self,a,b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

    def get_neighbor(self):
        for i in range(self.Data_size):
            point_i = self.Data[i]
            for j in range(i + 1, self.Data_size):
                point_j = self.Data[j]
                distance = self.get_distance(point_i, point_j)
                if distance <= self.Eps:
                    self.Neighbors[i].append(j)
                    self.Neighbors[j].append(i)

    def cluster(self):
        # a = {} print(1 in a) -> return false
        # check_dict = {}
        visit = [False for i in range(self.Data_size)]
        # in insert delete list is O(N) but queue is O(1) so i import queue
        # q.put(self.Core[0])

        for core_point in self.Core:
            if visit[core_point]: continue
            q = Queue()
            q.put(core_point)
            tmp = []
            while True:
                if q.qsize() == 0: break
                now = q.get()
                if now not in self.Core: continue

                for point in self.Neighbors[now]:
                    if visit[point]: continue
                    q.put(point)
                    tmp.append(point)
                    visit[point] = True

            self.Clusters.append(tmp)
        
        self.Clusters.sort(key = len)
        self.Clusters.reverse()
            


def main(input_f, n, eps, minpts):
    start = time.time()
    data = pd.read_csv(input_f, header = None, sep = '\t', names = ['key', 'x', 'y'])
    #data.plot.scatter(x = 'x', y = 'y')
    #plt.show(block = True)
    dbscan = DBSCAN(data ,int(n), int(eps), int(minpts), input_f[:-4])
    print(time.time() - start)

if __name__ == '__main__':
	argv=sys.argv

	main(argv[1],argv[2],argv[3],argv[4])