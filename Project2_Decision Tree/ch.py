#test.py
#data = pd.read_csv(input_name, sep='\t')

import numpy as np
import pandas as pd

a = pd.read_csv("dt_result1.txt", sep="\t")
b = pd.read_csv("dt_answer1.txt", sep="\t")

#print(a.columns)
#print(b.columns)

c = 0

for i in range(len(a)):
	#print(a["safety"], b["safety"])
	if a["car_evaluation"][i] == b["car_evaluation"][i]:
		c+=1

print(("%d"%c)+("/%d"%(len(a)))) 

