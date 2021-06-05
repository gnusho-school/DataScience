#python dt.py dt_train.txt dt_test.txt dt_result.txt
#Training file name=‘dt_train.txt’, test file name=‘dt_test.txt’, output file name=‘dt_result.txt’

import sys
import itertools
import numpy as np
import pandas as pd
from math import log

# pandas 함수들
# .nunique(): 각각의 attribute에 몇개의 unique값이 있는지 알려줌
# .unique(): 중복되는 값이 없는 열로 만들어 줌
# .value_counts(): 특정열의 value에 따른 갯수를 알 수
# list(data): column들을 알 수 있음
# venezuela = df[df['country'] == 'Venezuela'] 이런식으로 필터링 가능


class node:
	def __init__(self, attribute, val, CLASS = None):
		self.attribute = attribute
		self.val = val # 앞선 부모노드에서 나누어질 때 본인이 무슨 값이었는지 확인, None 이면 root 
		self.child = []
		self.CLASS = CLASS # leaf node가 아니라면 None값을 가집니다.
		self.zero_node = False # debugging을 위한 value로 majority voting을 경험했는지 나타냅니다.

class decision_tree:
	def __init__(self, train_data):
		self.train_data = train_data
		self.class_name = train_data.columns[-1]
		self.hi = train_data[self.class_name].value_counts().sort_values(ascending=False)
		self.root = self.build_tree_node(train_data, None)
		self.class_val = list(train_data)[len(list(train_data))-1]

	# get entropy function
	# Info(D) = -sigma(p[i]*log(p[i]))
	# pi = data중에서 i번째 class에 속할 확률
	def get_entropy(self, D):
		ret = 0.
		for c,v in D[self.class_name].value_counts().iteritems():
			add_val = 0.

			if v:
				p = v/len(D)
				add_val = p * log(p + (1e-9)) / log(2)

			ret += add_val

		return -ret 

	# get information gain function
	# gain(attribute) = info(D) - info<attribute>(D)
	# info<attribute>(D) = sigma(D[j] * info(D[j]))/len(D)
	def get_info_attr(self, D, attribute):
		attr = D[attribute].unique()
		info_a_d = 0.
		
		for a in attr:
			data_j = D[D[attribute] == a]
			info_a_d += len(data_j) * self.get_entropy(data_j)

		info_a_d/=len(D)

		return info_a_d

	# get node attribute by information gain
	def get_node_attr_by_info_gain(self, D):
		attribute = []

		for i in range(len(list(D))-1): 
			attribute.append(list(D)[i])

		info_d = self.get_entropy(D)
		node_attr = None
		max_gain = 0.

		for attr in attribute:
			info_attr = self.get_info_attr(D, attr)
			gain = info_d - info_attr
			if gain > max_gain :
				max_gain = gain
				node_attr = attr

		if max_gain < 0.2:
			node_attr = None

		return node_attr

	# split info a (D)=-sigman(Dj/D * log2(Dj/D))
	def get_split_info(self, D, attribute):
		attr = D[attribute].unique()
		split_info = 0.
		
		for a in attr:
			data_j = D[D[attribute] == a]
			val=len(data_j)/len(D)
			split_info+=val*log(val+1e-9,2)

		return - split_info

	# 각 attribute들의 value들의 갯수가 다르다면 그 경향성을 보완하는
	# gain ratio를 통해서 attribute를 구하도록 함
	def get_node_attr_by_gain_ratio(self, D):
		attribute = []

		for i in range(len(list(D))-1): 
			attribute.append(list(D)[i])

		info_d = self.get_entropy(D)
		node_attr = None
		max_gain_ratio = 0.

		for attr in attribute:
			info_attr = self.get_info_attr(D, attr)
			gain = info_d - info_attr
			split_info = self.get_split_info(D, attr)
			gain_ratio = gain / split_info
			if gain_ratio > max_gain_ratio :
				max_gain_ratio = gain_ratio
				node_attr = attr

		return node_attr

	# 더 이상 분류할 수 있는 label이 안남아있는데 class가 덜 분류됬을 때 사용한다.
	def majority_voting(self, D):
		major = D[self.class_name].value_counts().sort_values(ascending=False)
		ret = major.index[0]
		m = major[0]

		for i in range(1,len(major)):
			# sort values 함수의 특성으로 인해서 실행 때 마다 값이 달라지는 현상이 발생했다.
			# 그래서 이 정도 까지 왔다면 전체 data 개수를 봐서 더 작은 걸 사용하도록 했다
			if major[i] == m and self.hi[major.index[i]] < self.hi[ret]:
				ret = major.index[i]
			elif major[i] < m: break

		return ret

	def get_type(self,D):
		a = D[self.class_name].value_counts().sort_values(ascending=False)

		b = a[0]
		ret = True

		# ret가 True면 모든 value 수가 같으므로 info gain 사용
		# 아니면 gain ratio 사용

		for i in range(1,len(a)):
			if b != a[i]:
				ret = False
				break

		return ret

	# exec recursively
	# node들을 return 하도록 할 것임
	# child가 없으면 leaf node임
	def build_tree_node(self, D, value):

		if D[self.class_name].nunique() == 1:
			return node(None, value, D[self.class_name].unique()[0])

		if len(D.columns) == 1:
			return node(None, value, self.majority_voting(D))

		atype = self.get_type(D)

		# atype True면 모든 value 수가 같으므로 info gain 사용
		# 아니면 gain ratio 사용
		main_attr = None

		if atype: main_attr = self.get_node_attr_by_info_gain(D)
		else: main_attr = self.get_node_attr_by_gain_ratio(D)

		if main_attr == None:
			return node(None, value, self.majority_voting(D))

		nd = node(main_attr, value, None)

		for val in self.train_data[main_attr].unique():
			filtered_data = D[D[main_attr]== val]
			if len(filtered_data) > 0:
				nd.child.append(self.build_tree_node(filtered_data, val))
			elif len(filtered_data) == 0:
				nd.child.append(node(None, val, self.majority_voting(D)))

		return nd

	# test data를 class를 구한다.
	def get_CLASS(self, d):
		class_ret = None

		nd = self.root
		
		while True:

			if nd.CLASS:
				class_ret = nd.CLASS
				break

			for c in nd.child:
				if c.val == d[nd.attribute]:
					nd = c
					break
		
		return class_ret
		
	# test data를 받아서 decision tree를 test한다.
	def test_decision_tree(self, test_data):
		test_data[self.class_name] = None

		for d in range(len(test_data)):
			CLASS = self.get_CLASS(test_data.iloc[d])
			test_data[self.class_name][d] = CLASS

		return test_data 

	# Debugging을 위해서 만든 decision tree의 정보를 알려주는 함수
	def tree_bfs(self):
		print("########## Show Decision Tree Structure #########") 
		q = []
		q.append(self.root)

		while len(q):
			nd = q[0]
			q.pop(0)

			print(nd.attribute)
			if nd.attribute == None:
				print(nd.CLASS)
				print("\n")

			for c in nd.child:
				q.append(c)
				k = c.val
				p = c.attribute
				if c.val == None: k = "None"
				if c.attribute == None: p = "None"
				print(p + ": "+ k)
				if c.zero_node: print("majority_voting")

			print("**************")

		print("########## Show Decision Tree Structure Done #########\n") 

def file_read(input_name):
	data = pd.read_csv(input_name, sep='\t')
	return data

def main(train_f, test_f, output_f):
	train_data = file_read(train_f)
	dt = decision_tree(train_data)
	test_data = file_read(test_f)
	result = dt.test_decision_tree(test_data)
	result.to_csv(output_f, sep="\t")

if __name__ == '__main__':
	argv=sys.argv
	main(argv[1],argv[2],argv[3])