# coding: UTF-8
import os
import torch
from tqdm import tqdm
import time
import random
import numpy as np
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config,tokenizer):
    class_dict = {}
    for i, label in enumerate(config.class_list):
        class_dict[label] = i
    def load_dataset(path, seq_len=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                if config.label_map:
                    label = class_dict[label]
                token = tokenizer.tokenize(content)
                token = [CLS] + token
                mask = []
                token_ids = tokenizer.convert_tokens_to_ids(token)

                if seq_len:
                    if len(token) < seq_len:
                        mask = [1] * len(token_ids) + [0] * (seq_len - len(token))
                        token_ids += ([0] * (seq_len - len(token)))
                    else:
                        mask = [1] * seq_len
                        token_ids = token_ids[:seq_len]
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    train = load_dataset(config.train_path, config.sequence_len)
    dev = load_dataset(config.dev_path, config.sequence_len)
    test = load_dataset(config.test_path, config.sequence_len)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        random.shuffle(self.batches)  ### 随机打乱数据顺序
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class DatasetIterater_emotion(object):
    def __init__(self, batches, batch_size, tokenizer, device):
        self.device = device
        self.tokenizer = tokenizer

        batches_feature = batches["feature"]
        batches_text = batches["text"]
        batches_label = batches["labels"]
        data_list = []
        self.feature_tensor = []
        self.labels_tensor = []

        self.input_ids = torch.from_numpy(batches["input_ids"])
        self.token_type_ids = torch.from_numpy(batches["token_type_ids"])
        self.attention_mask = torch.from_numpy(batches["attention_mask"])

        #self.input_ids, self.token_type_ids, self.attention_mask = self._encode_fn(batches_text)

        self.feature_tensor = torch.from_numpy(batches_feature)
        self.labels_tensor = torch.from_numpy(batches_label)

        assert self.feature_tensor.shape[0] == self.labels_tensor.shape[0]

        self.batches_list = [j for j in range(self.feature_tensor.shape[0])]
        self.data_number = len(self.batches_list)

        self.batch_size = batch_size
        #self.batches = data_list
        random.shuffle(self.batches_list)  ### 随机打乱数据顺序
        self.n_batches = self.data_number // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if self.data_number % self.n_batches != 0:
            self.residue = True
        self.index = 0

    def _encode_fn(self, text_list):
        tokenizer = self.tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length= 128,
            return_tensors='pt'
        )
        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']
        return input_ids, token_type_ids, attention_mask

    def _to_tensor(self, batches_index):
        feature_tensor_one = [torch.unsqueeze(self.feature_tensor[i],dim=0) for i in batches_index]
        labels_tensor_one = [torch.unsqueeze(self.labels_tensor[i],dim=0) for i in batches_index]

        input_ids_tensor_one = [torch.unsqueeze(self.input_ids[i],dim=0) for i in batches_index]
        token_type_ids_tensor_one = [torch.unsqueeze(self.token_type_ids[i],dim=0) for i in batches_index]
        attention_mask_tensor_one = [torch.unsqueeze(self.attention_mask[i],dim=0) for i in batches_index]

        input_ids_batch = torch.cat(input_ids_tensor_one,dim=0).to(self.device)
        token_type_ids_batch =  torch.cat(token_type_ids_tensor_one,dim=0).to(self.device)
        attention_mask_batch = torch.cat(attention_mask_tensor_one, dim=0).to(self.device)
        feature = torch.cat(feature_tensor_one, dim=0).to(self.device)
        label_batch = torch.cat(labels_tensor_one, dim=0).to(self.device)

        input_ids_batch = input_ids_batch.type(torch.int64)
        token_type_ids_batch = token_type_ids_batch.type(torch.int64)
        attention_mask_batch = attention_mask_batch.type(torch.int64)
        feature_batch = feature.type(torch.float32)
        # pad前的长度(超过pad_size的设为pad_size)
        return input_ids_batch,token_type_ids_batch,attention_mask_batch,feature_batch,label_batch

    def _is_shuffle(self):
        ####  再次打乱顺序
        random.shuffle(self.batches_list)  ### 随机打乱数据顺序

    def __next__(self):
        if self.residue and self.index == self.n_batches:

            batches_index = self.batches_list[self.index * self.batch_size: self.data_number]
            self.index += 1
            batches = self._to_tensor(batches_index)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            #batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batches_index = self.batches_list[self.index * self.batch_size: (self.index + 1) * self.batch_size]

            self.index += 1
            batches = self._to_tensor(batches_index)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_emotion_iterator(dataset, config, tokenizer):
    iter = DatasetIterater_emotion(dataset, config.batch_size, tokenizer, config.device)
    return iter

###############################################################################################


class DatasetIterater_Mix(object):
    def __init__(self, data_list, batch_size, tokenizer, device, logger):
        self.device = device
        self.tokenizer = tokenizer

        feature_list = []
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        text_list = []
        label_list = []
        data_type = "list"
        if isinstance(data_list, list):
            for one_data in tqdm(data_list):
                feature_list.append(one_data["feature"])
                input_ids_list.append(one_data["input_ids"])
                token_type_ids_list.append(one_data["token_type_ids"])
                attention_mask_list.append(one_data["attention_mask"])
                text_list.append(one_data["text"])
                label_list.append(one_data["labels"])
            ####  置空数据，释放缓存
            data_list.clear()  # 置空列表,释放内存
            del data_list  # 删除列表,释放内存
            logger.info("####  释放缓存...")
            os.system("echo 1 > /proc/sys/vm/drop_caches")
            time.sleep(10)  ###  删除缓存后，睡眠10秒，让缓存进行释放
            logger.info("####  释放缓存完成。")
        elif isinstance(data_list, dict):
            feature_list = data_list["feature"]
            input_ids_list = data_list["input_ids"]
            token_type_ids_list = data_list["token_type_ids"]
            attention_mask_list = data_list["attention_mask"]
            text_list = data_list["text"]
            label_list = data_list["labels"]
            data_type = "numpy"
            ####  置空数据
            data_list.clear()  # 置空列表,释放内存
            del data_list  # 删除列表,释放内存
            logger.info("####  释放缓存...")
            os.system("echo 1 > /proc/sys/vm/drop_caches")
            time.sleep(10)  ###  删除缓存后，睡眠10秒，让缓存进行释放
            logger.info("####  释放缓存完成。")

        logger.info("###  数据加载中。。。")
        if data_type == "list":
            feature_numpy = np.array(feature_list)
            input_ids_numpy = np.array(input_ids_list)
            token_type_ids_numpy = np.array(token_type_ids_list)
            attention_mask_numpy = np.array(attention_mask_list)
            label_numpy = np.array(label_list)
            ####  置空数据
            feature_list.clear()  # 置空列表,释放内存
            input_ids_list.clear()  # 置空列表,释放内存
            token_type_ids_list.clear()  # 置空列表,释放内存
            attention_mask_list.clear()  # 置空列表,释放内存
            label_list.clear()  # 置空列表,释放内存
            del feature_list  # 删除列表,释放内存
            logger.info("####  释放缓存...")
            os.system("echo 1 > /proc/sys/vm/drop_caches")
            time.sleep(10)  ###  删除缓存后，睡眠10秒，让缓存进行释放
            logger.info("####  释放缓存完成。")

            self.input_ids = torch.from_numpy(input_ids_numpy)
            self.token_type_ids = torch.from_numpy(token_type_ids_numpy)
            self.attention_mask = torch.from_numpy(attention_mask_numpy)
            self.feature_tensor = torch.from_numpy(feature_numpy)
            self.labels_tensor = torch.from_numpy(label_numpy)
            ####  置空数据
            logger.info("####  置空数据")
            input_ids_numpy=token_type_ids_numpy=attention_mask_numpy=feature_numpy=label_numpy=[]
            feature_numpy.clear()  # 置空列表,释放内存
            input_ids_numpy.clear()  # 置空列表,释放内存
            token_type_ids_numpy.clear()  # 置空列表,释放内存
            attention_mask_numpy.clear()  # 置空列表,释放内存
            label_numpy.clear()  # 置空列表,释放内存
            del feature_numpy  # 删除列表,释放内存
            logger.info("####  释放缓存...")
            os.system("echo 1 > /proc/sys/vm/drop_caches")
            time.sleep(10)  ###  删除缓存后，睡眠10秒，让缓存进行释放
            logger.info("####  释放缓存完成。")
        elif data_type == "numpy":
            self.input_ids = torch.from_numpy(input_ids_list)
            self.token_type_ids = torch.from_numpy(token_type_ids_list)
            self.attention_mask = torch.from_numpy(attention_mask_list)
            self.feature_tensor = torch.from_numpy(feature_list)
            self.labels_tensor = torch.from_numpy(label_list)
            ####  置空数据
            del input_ids_list
            del token_type_ids_list
            del attention_mask_list
            del feature_list
            del label_list
            logger.info("####  释放缓存...")
            os.system("echo 1 > /proc/sys/vm/drop_caches")
            time.sleep(10)  ###  删除缓存后，睡眠10秒，让缓存进行释放
            logger.info("####  释放缓存完成。")
        logger.info("###  数据加载完成！")

        assert self.feature_tensor.shape[0] == self.labels_tensor.shape[0]

        self.batches_list = [j for j in range(self.feature_tensor.shape[0])]
        self.data_number = len(self.batches_list)
        logger.info("###  数据量：".format(self.data_number))
        logger.info("###  batch_size：{}".format(batch_size))

        self.batch_size = batch_size
        #self.batches = data_list
        random.shuffle(self.batches_list)  ### 随机打乱数据顺序
        self.n_batches = self.data_number // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if self.data_number % self.n_batches != 0:
            self.residue = True
        self.index = 0

    def _encode_fn(self, text_list):
        tokenizer = self.tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length= 128,
            return_tensors='pt'
        )
        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']
        return input_ids, token_type_ids, attention_mask

    def _to_tensor(self, batches_index):
        feature_tensor_one = [torch.unsqueeze(self.feature_tensor[i],dim=0) for i in batches_index]
        labels_tensor_one = [torch.unsqueeze(self.labels_tensor[i],dim=0) for i in batches_index]

        input_ids_tensor_one = [torch.unsqueeze(self.input_ids[i],dim=0) for i in batches_index]
        token_type_ids_tensor_one = [torch.unsqueeze(self.token_type_ids[i],dim=0) for i in batches_index]
        attention_mask_tensor_one = [torch.unsqueeze(self.attention_mask[i],dim=0) for i in batches_index]

        input_ids_batch = torch.cat(input_ids_tensor_one,dim=0).to(self.device)
        token_type_ids_batch =  torch.cat(token_type_ids_tensor_one,dim=0).to(self.device)
        attention_mask_batch = torch.cat(attention_mask_tensor_one, dim=0).to(self.device)
        feature = torch.cat(feature_tensor_one, dim=0).to(self.device)
        label_batch = torch.cat(labels_tensor_one, dim=0).to(self.device)

        input_ids_batch = input_ids_batch.type(torch.int64)
        token_type_ids_batch = token_type_ids_batch.type(torch.int64)
        attention_mask_batch = attention_mask_batch.type(torch.int64)
        feature_batch = feature.type(torch.float32)
        # pad前的长度(超过pad_size的设为pad_size)
        return input_ids_batch,token_type_ids_batch,attention_mask_batch,feature_batch,label_batch

    def _is_shuffle(self):
        ####  再次打乱顺序
        random.shuffle(self.batches_list)  ### 随机打乱数据顺序

    def __next__(self):
        if self.residue and self.index == self.n_batches:

            batches_index = self.batches_list[self.index * self.batch_size: self.data_number]
            self.index += 1
            batches = self._to_tensor(batches_index)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            #batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batches_index = self.batches_list[self.index * self.batch_size: (self.index + 1) * self.batch_size]

            self.index += 1
            batches = self._to_tensor(batches_index)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_emotion_mix_data_iterator(dataset, config, tokenizer):
    iter = DatasetIterater_Mix(dataset, config.batch_size, tokenizer, config.device)
    return iter



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def get_class_list(file_path):
    class_list = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            class_list.append(line.replace("\n", "").split("	")[1])
    class_list = list(set(class_list))
    return class_list

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
            line = line + "\n"
            f.write(line)
    f.close()