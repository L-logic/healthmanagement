from util.FileUtil import read_by_csv
import json
class GenTextModel:
	def __init__(self,ruler_path, dict_json) -> None:
		self.ruler_path = ruler_path
		self.dict_json = dict_json
		self.get_dict()
		self.get_status()
	def get_dict(self):
		with open(self.dict_json, 'r') as f:  
				self.fact_data = json.load(f)["fact"]
	
	def get_status(self):
		data = read_by_csv(self.ruler_path)
		i = 1
		res = []
		strs = ""
		for item in data.iloc[0]:
			strs =  strs + str(item)
			if i%3 == 0:
				res.append(strs)
				strs = ""
			i=i+1
		self.status = res
		return res
	
	def gen_text(self):
		# for i in range(self.status):
		sentences = []
		index = 0
		print(self.status[index])
		for item in ["PLR","BRN","ANN","RN","CS","LT","Delay"]:
			for xn in ["average","jitter","tendency"]:
				sentences.append(self.fact_data[item][self.status[index]])
				index=index+1
		# print(sentences)
		return sentences
