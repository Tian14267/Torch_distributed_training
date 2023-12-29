# coding: UTF-8
"""
@Author: fffan
@Time: 2023-12-29
@comment:
    安装 NCCL
    安装 horovod：0.28.1
        安装方法：HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
    使用：CUDA_VISIBLE_DEVICES="1,2,3" horovodrun -np 3 python run_fffan_horovod_distributed.py
    https://github.com/horovod/horovod/blob/master/docs/pytorch.rst
"""
import os
import copy
import json
import argparse
from utils import *
from sklearn import metrics
import torch.nn.functional as F
import torch.distributed as dist
from transformers import BertTokenizer
from optimization import BertAdam
from data_processer import DataProcess, ClassDataset
from summary import summary
from datetime import datetime
from logger import get_logger
from module_bert import BertConfig, BertTextModel
import torch.backends.cudnn as cudnn

import horovod.torch as hvd



def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model_name', type=str, default="emotion_models", help='')
    parser.add_argument('--bert_path', type=str,default="./data/bert_model_info", help='')
    parser.add_argument('--class_list', type=str, default=[], help='')
    parser.add_argument('--device', type=str, default="cuda", help='')

    parser.add_argument('--require_improvement', type=int, default=3000, help='')
    parser.add_argument('--num_epochs', type=int, default=20, help='')
    parser.add_argument('--num_classes', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--n_gpu', type=int, default=2, help="Changed in the execute process.")

    parser.add_argument('--learning_rate', type=float, default=4e-7, help='')
    parser.add_argument('--load_pkl', default=False, type=bool, help='')
    parser.add_argument('--is_train', default=True, type=bool, help='')
    parser.add_argument('--is_distributed', default=True, type=bool, help='')
    parser.add_argument('--is_batchsampler_in_traintrain', default=True, type=bool, help='训练阶段是否使用 batchsampler')
    parser.add_argument('--data_path', type=str, default="./data/cnews", help='')
    parser.add_argument('--class_list_path', type=str, default="./data/wav_text_emotion_data_mix_1206/label.json",
                        help='')
    parser.add_argument('--save_path', type=str, default="./saved_models/temp", help='')

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--master-port", default=0, type=str, help="distribted training")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()

    args.batch_size = int(args.batch_size / args.n_gpu)

    return args


def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    torch.cuda.set_device('cuda:{}'.format(0))

    n_gpu = torch.cuda.device_count()
    assert n_gpu == args.n_gpu
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    return device


def init_model_function(args, num_train_steps):
    config = BertConfig.from_json_file(args.bert_path)  # 加载bert模型配置信息
    config.num_labels = args.num_classes

    gpus = [i for i in range(args.n_gpu)]

    # Prepare model
    #model = BertTextModel.from_pretrained(args.bert_path, config=config).to(device)
    model = BertTextModel.from_pretrained(args.bert_path, config=config)
    model.cuda()

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    optimizer, model = prep_optimizer(args, model, num_train_steps)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    compression = hvd.Compression.fp16
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression)

    cudnn.benchmark = True

    return model, optimizer



def prep_dataloader(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    start_time = time.time()
    print("Loading data...")
    logger.info("Loading data...")
    dataProcess = DataProcess(args, args.data_path, tokenizer, 128, args.batch_size, logger)
    train_data_list, test_data_list, val_data_list,label_map = dataProcess.data_process()
    logger.info("Loading data Done!")

    # 1. datasets
    train_dataset = ClassDataset(
        train_data_list,
        'train'
    )

    test_dataset = ClassDataset(
        val_data_list,
        'valid'
    )
    val_dataset = ClassDataset(
        val_data_list,
        'valid'
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)


    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=2,
                                                  pin_memory=True,
                                                  sampler=test_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=2,
                                                  pin_memory=True,
                                                  sampler=val_sampler)


    ##########################################################################################################
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    logger.info('Time usage: {}'.format(time_dif))

    return train_dataloader, test_dataloader, val_dataloader, label_map


def prep_optimizer(args, model, num_train_steps):
    #if hasattr(model, 'module'):
    #    model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # """
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=0.05,
                         t_total=num_train_steps * args.num_epochs)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #"""
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    return optimizer, model


def reduce_mean(tensor):
    rt = tensor.clone()
    hvd.allreduce(rt, name='barrier')
    # # horovod.allreduce calculates the average value by default
    # # https://github.com/tczhangzhi/pytorch-distributed/issues/14
    # rt /= nprocs
    return rt

def train(args, model, optimizer, train_iter, dev_iter, test_iter, logger):
    start_time = time.time()
    model.train()

    dev_best_loss = float('inf')
    dev_best_acc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()

    num_batch = len(train_iter)
    save_path_in_epoch = ""
    for epoch in range(args.num_epochs):
        if not args.is_batchsampler_in_traintrain and args.is_distributed:
            ### DistributedSampler使用当前epoch作为随机数种子，从而使得不同epoch下有不同的shuffle结果。
            #### 所以，记得每次 epoch 开始前都要调 用一下 sampler 的 set_epoch 方法，这样才能让数据集随机 shuffle 起来。
            ####  注意：BatchSampler 没有该功能。
            train_iter.sampler.set_epoch(epoch)
            dev_iter.sampler.set_epoch(epoch)
        #print('Epoch [{}/{}]'.format(epoch + 1, args.num_epochs))
        logger.info('Epoch [{}/{}]'.format(epoch + 1, args.num_epochs))

        total_step = 0  # 记录进行到多少 step
        for i, batch in enumerate(train_iter):
            total_step += 1
            ####  数据转到 device
            batch = tuple(t.cuda(non_blocking=True) for t in batch)

            trains = batch[:-1]
            labels = batch[-1]
            outputs = model(trains[0],trains[1],trains[2])   ###  1: 神经网络forward过程

            true_label = labels.data.cpu()
            batch_pred = torch.max(outputs.data, 1)[1].cpu()
            train_acc = metrics.accuracy_score(true_label, batch_pred)

            #model.zero_grad()
            loss = F.cross_entropy(outputs, labels)   ### 2：获取loss，通过pred和label计算你损失函数
            if args.n_gpu > 1:
                #loss = loss.mean()  # mean() to average on multi-gpu.
                loss = reduce_mean(loss)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps


            optimizer.zero_grad()  ### 3： 清空网络中参数的梯度
            loss.backward()   ###  4： 反向传播，计算当前梯度
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  ####  梯度剪裁
            optimizer.step()  ###  5：根据梯度更新网络参数

            if hvd.rank() == 0 and total_step != 0 and total_step % 2 == 0 or total_step==num_batch:
                # 每多少轮输出在训练集和验证集上的效果
                dev_acc, dev_loss = evaluate(args, model, dev_iter)
                dev_acc = round(dev_acc, 5)
                #scheduler.step(dev_loss)

                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    save_path_in_epoch = os.path.join(args.save_path,args.model_name+"_pytorch_epoch_"+str(epoch)+".bin")
                    logger.info("### 模型Step:{0:>5}  最优ACC：{1:>5}   保存模型：{2}".format(total_step,dev_acc,save_path_in_epoch))
                    torch.save(model.state_dict(), save_path_in_epoch)
                    improve = '*'
                    last_improve = total_step
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)


                ####  打印前对数据进行处理
                loss_print = loss.clone().detach().cpu().numpy()
                dev_loss_print = dev_loss.clone().detach().cpu().numpy()
                msg = 'Epoch: {0:>3}, Iter: {1:>5}/{2:>5}, Start learing_rate:{3:>6} ,Train Loss: {4:>5.6},  Train Acc: {5:>6.3%},  Val Loss: {6:>5.6},  Val Acc: {7:>6.3%},  Time: {8} {9}  Rank：{10}'
                #logger.info(msg.format(epoch,total_step,num_batch,optimizer.defaults['lr'], loss_print, train_acc, dev_loss_print, dev_acc, time_dif, improve, hvd.rank()))###  方式一
                logger.info(msg.format(epoch, total_step, num_batch, args.learning_rate, loss_print, train_acc, dev_loss_print,dev_acc, time_dif, improve, hvd.rank()))###  方式二
                model.train()

            if total_step - last_improve > args.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break

        if not save_path_in_epoch:
            save_path_in_epoch = os.path.join(args.save_path, args.model_name + "_pytorch_epoch_" + str(epoch) + ".bin")
            logger.info("### 模型 Epoch:{0}  保存模型：{1}".format(epoch, save_path_in_epoch))
            torch.save(model.state_dict(), save_path_in_epoch)

        if flag:  # 记录是否很久没有效果提升
            break

    if save_path_in_epoch:
        test(args, model, test_iter,save_path_in_epoch,  logger)



def test(config, model, test_iter,model_path,  logger):
    # test
    model.load_state_dict(torch.load(model_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    logger.info(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    logger.info("Precision, Recall and F1-Score...")
    print(test_report)
    logger.info(test_report)
    print("Confusion Matrix...")
    logger.info("Confusion Matrix...")
    print(test_confusion)
    logger.info(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage: {}".format(time_dif))
    logger.info("Time usage: {}".format(time_dif))



def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            batch = tuple(t.cuda(non_blocking=True) for t in batch)
            texts = batch[:-1]
            labels = batch[-1]
            outputs = model(texts[0],texts[1],texts[2])
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def main():
    args = get_args()
    ###################################
    hvd.init()
    args.local_rank = hvd.local_rank()
    torch.cuda.set_device(args.local_rank)


    ###################################

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    global logger
    if args.save_path == "":
        args.save_dir = os.path.join(os.path.curdir, args.save_path, "training", datetime.now(
        ).strftime("%Y-%d-%m-%H-%M-%S-") + str(random.random())[2:6])
        os.makedirs(args.save_path, exist_ok=True)
    logger = get_logger("info", args.save_path + '/training.log')
    logger.info(args)

    ###############
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    train_dataloader, test_dataloader, val_dataloader, label_map = prep_dataloader(args)
    ##############################################################################################

    args.class_list = list(label_map.keys())
    args.num_classes = len(args.class_list)

    # train
    args.n_class = args.num_classes  # 设置分类模型的输出个数
    num_train_steps = len(train_dataloader)

    model, optimizer = init_model_function(args, num_train_steps )


    if args.is_train:
        #####
        #summary_input = [(128,),(128,),(128,)]
        #summary(model=model, input_size=summary_input, batch_size=args.batch_size, device=device_str, logger=logger)
        train(args, model,optimizer, train_dataloader, val_dataloader, test_dataloader, logger)

    else:
        #summary_input = [(128,), (128,), (128,)]
        #summary(model=model, input_size=summary_input, batch_size=args.batch_size, device=device_str)
        #model_path = "./saved_models/wav_emotion_models_1116_3/emotion_models_pytorch_model_14.bin"
        model_path = os.path.join(args.save_path,"emotion_models_pytorch_epoch_1.bin")
        test(args, model, test_dataloader, model_path, device, logger)


if __name__ == '__main__':
    main()
