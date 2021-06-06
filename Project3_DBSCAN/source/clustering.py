# -*- coding: utf-8 -*-
#python clustering.py data-3/input1.txt 8 15 22
#python clustering.py input1.txt 8 15 22
import sys
import itertools
import numpy as np
import pandas as pd
import time
from queue import Queue #put, get

class DBSCAN:
    def __init__(self, data, n, eps, minpts, input_f):
        self.input_string = input_f
        self.N = n 
        self.Eps = eps
        self.Minpts = minpts

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

        self.Clusters = []
        self.cluster()

        self.write_clusters()

    def write_clusters(self):
        for i in range(self.N):
            file_name = self.input_string + "_cluster_" + str(i) + ".txt"
            f = open(file_name, 'w')
            self.Clusters[i].sort()
            for point in self.Clusters[i]:
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
                # for문 안에서 계속 확인하는 것보다 이게 나을것이라 생각함

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
    dbscan = DBSCAN(data ,int(n), int(eps), int(minpts), input_f[:-4])
    print(time.time() - start)

if __name__ == '__main__':
	argv=sys.argv
	main(argv[1],argv[2],argv[3],argv[4])