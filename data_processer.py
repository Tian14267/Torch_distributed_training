# coding: UTF-8
"""
@Author: fffan
@Time: 2023-12-29
"""
import os
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset




class NewsDataset(Dataset):
	def __init__(self, input_ids, token_type_ids, attention_mask, mfcc_feature, labels=None):
		super().__init__()
		self.input_ids = input_ids
		self.token_type_ids = token_type_ids
		self.attention_mask = attention_mask
		self.labels = labels
		self.features = mfcc_feature

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, index):
		a = self.input_ids[index]
		b = self.token_type_ids[index]
		c = self.attention_mask[index]
		d = self.labels[index]
		f = self.features[index]
		#return {"input_ids": a, "token_type_ids": b, "attention_mask": c, "labels": d,"features":f}
		return {"input_ids": a, "token_type_ids": b, "attention_mask": c, "labels": d}


def read_files(file):
	with open(file,"r",encoding="utf-8") as f:
	#with open(file, "r", encoding="GBK") as f:
		lines = f.readlines()
		lines_out = []
		for line in lines:
			line = line.replace("\n","")
			lines_out.append(line)
	f.close()
	return lines_out

def write_file(lines,file):
	with open(file,"w",encoding="utf-8") as f:
		for line in lines:
			line = str(line).replace("\'","\"") + "\n"
			f.write(line)
	f.close()




class DataProcess:
	"""
	数据处理模块
	"""
	def __init__(self, args, data_path, tokenizer, seq_max_len=128, batch_size=32, logger=None, label_map=None):
		self.args = args
		self.data_path = data_path
		self.tokenizer = tokenizer
		self.seq_max_len = seq_max_len
		self.batch_size = batch_size
		self.label_map = label_map
		self.logger = logger


	def load_data(self, path):
		#####  Train data
		lines = read_files(os.path.join(path,"train.txt"))
		train_data_dict = {}
		for i, line in enumerate(lines):
			label, text = line.split("\t")
			if label not in train_data_dict.keys():
				train_data_dict[label] = [text]
			else:
				train_data_dict[label] = train_data_dict[label] + [text]

		#####  Test data
		lines = read_files(os.path.join(path, "test.txt"))
		test_data_dict = {}
		for i, line in enumerate(lines):
			label, text = line.split("\t")
			if label not in test_data_dict.keys():
				test_data_dict[label] = [text]
			else:
				test_data_dict[label] = test_data_dict[label] + [text]

		#####  Val data
		lines = read_files(os.path.join(path, "val.txt"))
		val_data_dict = {}
		for i, line in enumerate(lines):
			label, text = line.split("\t")
			if label not in val_data_dict.keys():
				val_data_dict[label] = [text]
			else:
				val_data_dict[label] = val_data_dict[label] + [text]

		if self.label_map is None:
			self.label_map = {}
			for i,label in enumerate(list(set(train_data_dict.keys()))):
				self.label_map[label] = i

			#with open(os.path.join(self.args.pkl_data_files, "label.json"), 'w') as writer:
			#	writer.write(json.dumps(self.label_map, ensure_ascii=False) + '\n')

		#labels_id = [self.label_map[one_label] for one_label in labels]

		return train_data_dict, test_data_dict, val_data_dict, self.label_map


	def encode_fn(self, text_list):
		tokenizer = self.tokenizer(
			text_list,
			padding="max_length",
			truncation=True,
			max_length=self.seq_max_len,
			return_tensors='pt'
		)
		input_ids = tokenizer['input_ids'].numpy()
		token_type_ids = tokenizer['token_type_ids'].numpy()
		attention_mask = tokenizer['attention_mask'].numpy()
		return input_ids, token_type_ids, attention_mask

	def process_data(self,  text_list):
		input_ids, token_type_ids, attention_mask = self.encode_fn(text_list)

		input_ids_np = input_ids.numpy()
		token_type_ids_np = token_type_ids.numpy()
		attention_mask_np = attention_mask.numpy()

		return input_ids_np,token_type_ids_np,attention_mask_np


	def data_process(self):
		if not self.args.load_pkl:
			data_train, data_test, data_val, label_map = self.load_data(self.data_path)  # 加载训练数据集

			data_list_train = []
			for key in tqdm(data_train.keys(),desc="# Train"):
				key_id = label_map[key]  ####  把 label转换成 id
				for one_text in tqdm(data_train[key],desc="## {}".format(key)):
					input_ids, token_type_ids, attention_mask = self.encode_fn(one_text)
					one_data = {"input_ids":np.squeeze(input_ids, axis=0),
								"token_type_ids":np.squeeze(token_type_ids, axis=0),
								"attention_mask":np.squeeze(attention_mask, axis=0),
								"label":key_id
								}
					data_list_train.append(one_data)
			######

			data_list_test = []
			for key in tqdm(data_test.keys(),desc="# Test"):
				key_id = label_map[key]  ####  把 label转换成 id
				for one_text in tqdm(data_test[key],desc="## {}".format(key)):
					input_ids, token_type_ids, attention_mask = self.encode_fn(one_text)
					one_data = {"input_ids": np.squeeze(input_ids, axis=0),
								"token_type_ids": np.squeeze(token_type_ids, axis=0),
								"attention_mask": np.squeeze(attention_mask, axis=0),
								"label": key_id
								}
					data_list_test.append(one_data)
			######
			data_list_val = []
			for key in tqdm(data_val.keys(),desc="# Dev"):
				key_id = label_map[key]  ####  把 label转换成 id

				for one_text in tqdm(data_val[key],desc="## {}".format(key)):
					input_ids, token_type_ids, attention_mask = self.encode_fn(one_text)
					one_data = {"input_ids": np.squeeze(input_ids, axis=0),
								"token_type_ids": np.squeeze(token_type_ids, axis=0),
								"attention_mask": np.squeeze(attention_mask, axis=0),
								"label": key_id
								}
					data_list_val.append(one_data)

			#################################################################
			os.makedirs(os.path.join(self.args.data_path, "./pkl_data"), exist_ok=True)
			###
			data_save_train_dict = {}
			data_save_train_dict["data"] = data_list_train
			with open(os.path.join(self.args.data_path, "./pkl_data/train.pkl"), 'wb') as f:
				pickle.dump(data_save_train_dict, f)

			data_save_test_dict = {}
			data_save_test_dict["data"] = data_list_test
			with open(os.path.join(self.args.data_path, "./pkl_data/test.pkl"), 'wb') as f:
				pickle.dump(data_save_test_dict, f)

			data_save_dev_dict = {}
			data_save_dev_dict["data"] = data_list_val
			with open(os.path.join(self.args.data_path, "./pkl_data/val.pkl"), 'wb') as f:
				pickle.dump(data_save_dev_dict, f)

			with open(os.path.join(self.args.data_path, "./pkl_data/label.json"), 'w') as writer:
				writer.write(json.dumps(label_map, ensure_ascii=False) + '\n')

		else:
			with open(os.path.join(self.args.data_path, "./pkl_data/train.pkl"), 'rb') as f:
				data_save_train_dict = pickle.load(f)
			data_list_train = data_save_train_dict["data"]
			#self.logger.info('完成训练数据加载')
			with open(os.path.join(self.args.data_path, "./pkl_data/test.pkl"), 'rb') as f:
				data_save_val_dict = pickle.load(f)
			data_list_test = data_save_val_dict["data"]
			#self.logger.info('完成测试数据加载')
			with open(os.path.join(self.args.data_path, "./pkl_data/val.pkl"), 'rb') as f:
				data_save_val_dict = pickle.load(f)
			data_list_val = data_save_val_dict["data"]
			#self.logger.info('完成测试数据加载')
			with open(os.path.join(self.args.data_path, "./pkl_data/label.json"), 'r', encoding="utf-8") as load_f:
				label_map = json.load(load_f)

		###################################################################
		print('训练集数量{}：格式{}，验证集：{}'.format(len(data_list_train), data_list_train[0]["input_ids"].shape, len(data_list_val)))
		#self.logger.info('训练集数量{}：格式{}，测试集：{}'.format(len(data_list_train), data_list_train[0]["input_ids"].shape, len(data_list_val)))

		return data_list_train, data_list_test, data_list_val, label_map



class ClassDataset(Dataset):
	def __init__(self, data_info, data_type):
		self.data_type = data_type
		self.input_ids_list=[]
		self.token_type_ids_list=[]
		self.attention_mask_list=[]
		self.label_list=[]
		for one_data in tqdm(data_info):
			self.input_ids_list.append(one_data["input_ids"])
			self.token_type_ids_list.append(one_data["token_type_ids"])
			self.attention_mask_list.append(one_data["attention_mask"])
			self.label_list.append(one_data["label"])
		print("#####################")


	def _get_text(self, index):
		text = self.input_ids_list[index]  ###  [batch, 128]
		text_type = self.token_type_ids_list[index] ###  [batch, 128]
		text_mask = self.attention_mask_list[index] ###  [batch, 128]

		return text, text_type, text_mask


	def _get_labels(self, index):
		label_id = self.label_list[index]

		return label_id


	def __len__(self):
		return len(self.label_list)

	def __getitem__(self, index):
		text_id, text_type, text_mask = self._get_text(index)
		label = self._get_labels(index)

		return text_id, text_type, text_mask, label



