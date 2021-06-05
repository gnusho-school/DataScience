# -*- coding: utf-8 -*-
import sys
import itertools
import numpy as np

#File read and write functions
def file_read(input_name):
	all_items=set()
	transactions=[]
	f=open(input_name,'r')

	while True:
		transaction_tmp=[]
		line=f.readline()
		if not line: break
		transaction_tmp=line.strip().split('\t')
		
		set_tmp=set()
		str2int=[]
		
		for i in transaction_tmp:
			num=int(i)
			str2int.append(num)

		set_tmp.update(str2int)
		all_items=all_items|set_tmp
		transactions.append(str2int)
	f.close()
	return transactions,all_items

#implemnetation of apriori algorithm
class apriori:
	def __init__(self, min_s, transactions_tmp, all_items):
		self.transactions=transactions_tmp #each transaction data type is set
		self.dsize=len(self.transactions)
		self.min_support=float(min_s)
		self.minSnum=(self.min_support*self.dsize)/100.0
		self.freq_pats=[]
		self.items=all_items
	
	#get support from transactions for candidate
	def get_support(self, candidate):
		cnt=0
		for i in self.transactions:
			transaction=set(i)
			if candidate.issubset(transaction): cnt+=1
		return cnt
	
	#get candidates from befroe_patterns ,which have k elements 
	def get_candidates(self,before_patterns,k):
		candidates=[]
		end=len(before_patterns)
		self_join=before_patterns[0]
		for i in range(1,end):
			self_join=self_join|before_patterns[i]

		k_subsets=map(set, itertools.combinations(self_join, k))
		for k_subset in k_subsets:
			is_candidate=True
			subset=map(set,itertools.combinations(k_subset,k-1))
			for s in subset:
				if s in before_patterns: continue
				is_candidate=False
				break
			if is_candidate: candidates.append(k_subset)
			
		return candidates

	def get_freq_items(self):
		item_support={}
		for i in self.items:
			item_support[i]=0
		for transaction in self.transactions:
			for item in transaction:
				item_support[item]+=1
		freq_items=[]
		for item in self.items:
			if item_support[item]>= self.minSnum:
				tmp=[]
				tmp.append(item)
				freq_items.append(set(tmp))
		return freq_items

	def get_freq_pat(self):
		before_patterns=self.get_freq_items()
		for k in range (2, len(self.items)+1):
			after_patterns=[]
			candidates=self.get_candidates(before_patterns,k)
			if len(candidates)==0: break
			for c in candidates:
				c_support=self.get_support(c)
				if c_support>=self.minSnum:
					self.freq_pats.append(c)
					after_patterns.append(c)
			if len(after_patterns)==0: break
			before_patterns=after_patterns
			#print(k,len(candidates),len(after_patterns))

	def get_association_patterns(self):
		self.get_freq_pat()
		association_patterns=[]
		#print(len(self.freq_pats))
		for pattern in self.freq_pats:
			support_num=self.get_support(pattern)
			support=round((support_num*100.0)/self.dsize,2)
			for a_len in range(1,len(pattern)):
				a_set=map(set,itertools.combinations(pattern, a_len))
				for a in a_set:
					a_support_num=self.get_support(a)
					confidence=round(support_num*100.0/a_support_num,2)
					association_patterns.append([a,pattern-a,support,confidence])
		return association_patterns

def set2write(set_input):
	l=list(set_input)
	ret='{'
	for i in range(len(l)-1):
		ret+='%d,'%l[i]
	ret+='%d}\t'%l[len(l)-1]
	return ret

def main(min_s, input_f, output_f):
	transactions_tmp,all_items=file_read(input_f)
	apr=apriori(min_s,transactions_tmp,all_items)
	association_patterns=apr.get_association_patterns()

	f=open(output_f,'w')
	for pattern in association_patterns:
		A=set2write(pattern[0])
		B=set2write(pattern[1])
		support='%.2f\t'%pattern[2]
		confidence='%.2f\n'%pattern[3]
		line=A+B+support+confidence
		f.write(line)
	f.close()

if __name__ == '__main__':
	argv=sys.argv
	main(argv[1],argv[2],argv[3])