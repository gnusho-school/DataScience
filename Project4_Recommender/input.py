# -*- coding: utf-8 -*-
# python rmse.py u1
import sys
import itertools
import numpy as np
import pandas as pd
import time
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from sklearn.metrics import mean_squared_error

class Recommender:
    def __init__(self, base_f, test_f):
        
        self.non_empty_count = 0
        self.every_cnt = 0
        self.rating_cnt = np.zeros((5))

        # matrix 만들기
        self.test_file_name = test_f
        self.base_file_name = base_f
        self.user_item_matrix = self.base_file_read_create_matrix()

        print(np.sum(self.user_item_matrix) / self.non_empty_count)
        print(self.rating_cnt)

        # matrix factorization 구현
        #P, Q = self.matrix_factorization()
        #self.user_item_matrix_tr = np.dot(P,Q.T)

        # output file 만들기
        #self.test_file_read_create_output()

    def matrix_factorization(self, K= 5, steps= 10, alpha=0.002, beta=0.02):
        '''
        R: rating matrix
        P: |U| * K (User features matrix)
        Q: |D| * K (Item features matrix)
        K: latent features
        steps: iterations
        alpha: learning rate
        beta: regularization parameter'''

        # alpha -> 0.0002 => 0.002 causes better result in 10 steps // go to 100 steps
        R = self.user_item_matrix
        N = len(self.user_item_matrix)
        M = len(self.user_item_matrix[0])
        #P = np.random.rand(N,K) * 5
        #print(np.mean(P))
        P = np.zeros((N, K)) + (0.5)
        #Q = np.random.rand(M,K) * 5
        #print(np.mean(Q))
        Q = np.zeros((M, K)) + (0.5)

        Q = Q.T

        for step in range(steps + 1):
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        # calculate error
                        eij = R[i][j] - np.dot(P[i,:],Q[:,j])

                        for k in range(K):
                            # calculate gradient with a and beta parameter
                            #P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                            #Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

                             P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j])
                             Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k])
                            # No regularization causes overfitting problem maybe
                            # it makes higher rmse result

            eR = np.dot(P,Q)

            e = 0

            for i in range(len(R)):

                for j in range(len(R[i])):

                    if R[i][j] > 0:
                        self.non_empty_count += 1
                        e = e + pow(R[i][j] - eR[i][j], 2)

                        for k in range(K):

                            e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
            # 0.001: local minimum

            e = e / self.non_empty_count

            if step % 10 == 0: print(step, e)
            
            #if e < 0.009: break

        return P, Q.T

    def test_file_read_create_output(self):

        f = open(self.test_file_name, 'r')
        f_prediction = open(self.base_file_name + '_prediction.txt', 'w')

        while True:
            line = f.readline()
            if not line: break
            tmp = line[:-1].split('\t')
            tmp_int = [int(x) for x in tmp]
            #user\t item\t rating\t timestamp\n
            
            predict_rating = self.user_item_matrix_tr[tmp_int[0] - 1][tmp_int[1] - 1]
            f_prediction.write(tmp[0] + '\t' + tmp[1] + '\t' + str(predict_rating) + '\n') 

        f.close()
        f_prediction.close()

    def base_file_read_create_matrix(self):
        max_user_id = 0
        max_item_id = 0
        
        # 1차 file read 시작, 최대 id값들 받아올 예정
        f = open(self.base_file_name, 'r')

        while True:
            line = f.readline()
            if not line: break
            tmp = line[:-1].split('\t')
            #user\t item\t rating\t timestamp\n
            max_user_id = max(max_user_id, int(tmp[0]) )
            max_item_id = max(max_item_id, int(tmp[1]) )

        f.close()

        f = open(self.test_file_name, 'r')
        while True:
            line = f.readline()
            if not line: break
            tmp = line[:-1].split('\t')
            #user\t item\t rating\t timestamp\n
            max_user_id = max(max_user_id, int(tmp[0]) )
            max_item_id = max(max_item_id, int(tmp[1]) )

        f.close()

        # 1차 file read 끝, empty matrix 만들고 다시 한 번 읽을 것임
        print(max_user_id, max_item_id)
        self.every_cnt = max_item_id * max_user_id
        # np empty array 만들기

        user_item_matrix = np.empty((max_user_id, max_item_id), float)
        
        # 2차 file read 시작, matrix 완성하고 return 할 예정
        f = open(self.base_file_name, 'r')

        while True:
            line = f.readline()
            if not line: break
            tmp = line[:-1].split('\t')
            # user\t item\t rating\t timestamp\n
            tmp = [int(x) for x in tmp]
            
            # 메모리 낭비 줄이기 위해서 user와 item id 모두 1을 뺀 값 사용함
            # 마지막에 처리해 줄 것
            user_item_matrix[tmp[0] - 1][tmp[1] - 1] = tmp[2]

            if tmp[2] != 0:
                self.non_empty_count += 1
                self.rating_cnt[tmp[2] - 1] += 1

        f.close()

        return user_item_matrix

def main(base, test):
    start = time.time()
    recommender = Recommender(base, test)
    print('processing time: %f'%(time.time() - start))

if __name__ == '__main__':
	argv=sys.argv
	main(argv[1],argv[2])