import json
import os
import numpy as np
from util.FileUtil import read_by_csv
"""
规则过滤模型
"""
class RulerModel:
	"""
	"""
	def __init__(self,math_data,ruler_json,init=False):
			self.data = None
			self.ruler_json = ruler_json
			self.data_path = math_data
			self.filename = self.data_path.split('\\')[-1]
			data = read_by_csv(self.data_path)
			self.data = np.array(data, dtype=float)
			# if init==True:
			# 		self.build_folder()
	
	def build_folder():
		pass
	
	def process_data(self):
		res = []
		for i in range(len(self.data[0])):
			attribute = self.compute_attribute(i)
			res.append(attribute)
		self.math_data = res
		return res
	
	def compute_attribute(self,index):
		avg = self.data[:,index].mean()
		column_data = self.data[:, index]
		diffs = column_data[1:] - column_data[:-1]
		jitter = np.mean(diffs) if diffs.size > 0 else np.nan
		variance = np.var(self.data[:, index]) 
		return [avg,jitter,variance]
	
	def get_status(self):
		with open(self.ruler_json, 'r') as f:  
				ruler = json.load(f)
		rule_dict = {"ruler_node":["PLR","BEN","ANN","RN","CS"],"ruler_link":["PLR","BEN","Delay","LT"]}
		i = 0
		res = []
		# [[],[]]
		for thing in ["ruler_node","ruler_link"]:
			for item in rule_dict[thing]:
				w = 0
				att = []
				# print(len(ruler[thing][item]))
				for times in ruler[thing][item]:
					status = 0
					for index in range(len(times)-1):
						if self.math_data[i][w] < times[0]:
							att.append(0)
							break
						if self.math_data[i][w] >= times[index] and self.math_data[i][w] <= times[index+1]:
							att.append(status+1)
							break
						if self.math_data[i][w] > times[-1]:
							att.append(-1)
							break
						status = status+1
					w = w+1
				i=i+1
				res.append(att)
		return res