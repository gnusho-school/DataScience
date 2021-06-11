# -*- coding: utf-8 -*-
# python rmse.py u1
import sys
import itertools
import numpy as np
import pandas as pd
import time

def main(base):
    prediction = base + '.base_prediction.txt'
    test = base + '.test'

    f_prediction = open(prediction, 'r')
    f_test = open(test,'r')

    predicttion_list = []
    test_list = []
    n = 0

    while True:
        line_prediction = f_prediction.readline()
        if not line_prediction: break

        line_test = f_test.readline()
        if not line_test: break

        n += 1

        tmp_prediction = line_prediction[:-1].split('\t')
        tmp_test = line_test[:-1].split('\t')

        predicttion_list.append(float(tmp_prediction[2]))
        test_list.append(float(tmp_test[2]))

    predicttion_list = np.array(predicttion_list)
    test_list = np.array(test_list)

    subtract = predicttion_list - test_list
    #print(subtract)

    rmse = np.sqrt(np.sum(np.power(subtract, 2)) / n)

    print(rmse)

if __name__ == '__main__':
	argv=sys.argv
	main(argv[1])