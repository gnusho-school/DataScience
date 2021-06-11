# -*- coding: utf-8 -*-
# python recommender.py u1.base u1.test
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

        # matrix 만들기
        self.test_file_name = test_f
        self.base_file_name = base_f
        self.user_item_matrix = self.base_file_read_create_matrix()

        # matrix factorization 구현
        user_matrix, item_matrix = self.matrix_factorization()
        self.user_item_matrix_hat = np.dot(user_matrix,item_matrix.T)

        # output file 만들기
        self.test_file_read_create_output()

    def matrix_factorization(self):
        K = 5
        epochs = 20
        alpha = 0.001
        beta = 0.02

        R = self.user_item_matrix
        N = len(self.user_item_matrix)
        M = len(self.user_item_matrix[0])

        default = 0.75
        print(default)

        user_matrix = np.zeros((N, K)) + default
        item_matrix = np.zeros((M, K)) + default
        item_matrix = item_matrix.T

        for epoch in range(epochs + 1):
            E = R - np.dot(user_matrix, item_matrix)
            
            for i in range(N):
                for j in range(M):
                    if R[i][j] > 0:
                        # calculate error
                        eij = R[i][j] - np.dot(user_matrix[i,:],item_matrix[:,j])

                        for k in range(K):
                            user_matrix[i][k] = user_matrix[i][k] + alpha * (2 * eij * item_matrix[k][j])
                            item_matrix[k][j] = item_matrix[k][j] + alpha * (2 * eij * user_matrix[i][k])
                            
            eR = R - np.dot(user_matrix,item_matrix)

            e = 0

            for i in range(N):
                for j in range(M):
                    if R[i][j] > 0:
                        e = e + pow(eR[i][j], 2)

                        for k in range(K):
                            e = e + (beta/2) * (pow(user_matrix[i][k],2) + pow(item_matrix[k][j],2))

            e = e / self.non_empty_count

            if epoch % 5 == 0: print(epoch, e)

        return user_matrix, item_matrix.T

    def test_file_read_create_output(self):

        f = open(self.test_file_name, 'r')
        f_prediction = open(self.base_file_name + '_prediction.txt', 'w')

        while True:
            line = f.readline()
            if not line: break
            tmp = line[:-1].split('\t')
            tmp_int = [int(x) for x in tmp]
            #user\t item\t rating\t timestamp\n
            
            predict_rating = self.user_item_matrix_hat[tmp_int[0] - 1][tmp_int[1] - 1]
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
            self.non_empty_count += 1
            user_item_matrix[tmp[0] - 1][tmp[1] - 1] = tmp[2]

        f.close()

        return user_item_matrix

def main(base, test):
    start = time.time()
    recommender = Recommender(base, test)
    print('processing time: %f'%(time.time() - start))

if __name__ == '__main__':
	argv=sys.argv
	main(argv[1],argv[2])