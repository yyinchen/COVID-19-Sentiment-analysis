import torch
import os
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel
import codecs
from config import get_config
from data import load_dataset
from model import Transformer, Gru_Model, BiLstm_Model, Lstm_Model, Rnn_Model, TextCNN_Model, Transformer_CNN_RNN, \
    Transformer_Attention, Transformer_CNN_RNN_Attention
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd

TRANSFORMERS_OFFLINE=1
class Niubility:
    def __init__(self,args, logger):
        self.args = args

        args.model_name = 'bert'  # 打印出模型
        args.method_name = 'lstm+textcnn'  # 打印出方法
        args.num_epoch = 2    # 打印出训练次数

        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))  # 加载模型
        # Create model
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.input_size = 768
            base_model = AutoModel.from_pretrained('bert-base-uncased')

        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            self.input_size = 768
            base_model = AutoModel.from_pretrained('roberta-base')
        else:
            raise ValueError('unknown model')
        # Operate the method
        if args.method_name == 'fnn':
            self.Mymodel = Transformer(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'gru':
            self.Mymodel = Gru_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'lstm':
            self.Mymodel = Lstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'bilstm':
            self.Mymodel = BiLstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'rnn':
            self.Mymodel = Rnn_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'textcnn':
            self.Mymodel = TextCNN_Model(base_model, args.num_classes)
        elif args.method_name == 'attention':
            self.Mymodel = Transformer_Attention(base_model, args.num_classes)
        elif args.method_name == 'lstm+textcnn':
            self.Mymodel = Transformer_CNN_RNN(base_model, args.num_classes)
        elif args.method_name == 'lstm_textcnn_attention':
            self.Mymodel = Transformer_CNN_RNN_Attention(base_model, args.num_classes)
        else:
            raise ValueError('unknown method')

        self.Mymodel.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0

        self.Mymodel.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            predicts = self.Mymodel(inputs)
            loss = criterion(predicts, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
            n_train += targets.size(0)

        return train_loss / n_train, n_correct / n_train

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        TP, TN, FP, FN = 0, 0, 0, 0

        self.Mymodel.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)
                # 保存真实数据和预测数据
                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)

                for i in range(len(predicts)):
                    y_predict = predicts[i].detach().numpy().tolist()
                    y_target = targets[i].detach().numpy().tolist()

                    y_predict_str1 = str(y_predict[0])
                    y_predict_str2 = str(y_predict[1])
                    y_target_str = str(y_target)

                    output = '{}\t{}\t{}\n'.format(y_predict_str1, y_predict_str2,y_target_str)
                    print(output)


                ground_truth = targets
                predictions = torch.argmax(predicts, dim=1)
                TP += torch.logical_and(predictions.bool(), ground_truth.bool()).sum().item()
                FP += torch.logical_and(predictions.bool(), ~ground_truth.bool()).sum().item()
                FN += torch.logical_and(~predictions.bool(), ground_truth.bool()).sum().item()
                TN += torch.logical_and(~predictions.bool(), ~ground_truth.bool()).sum().item()
            if TP + FP > 0 and TP + FN > 0:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)

                f1 = 2 * precision * recall / (precision + recall)

        return test_loss / n_test, n_correct / n_test, precision, recall, f1

    def run(self):
        train_dataloader, test_dataloader = load_dataset(tokenizer=self.tokenizer,
                                                         train_batch_size=self.args.train_batch_size,
                                                         test_batch_size=self.args.test_batch_size,
                                                         model_name=self.args.model_name,
                                                         method_name=self.args.method_name,
                                                         workers=self.args.workers)
        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        l_acc, l_trloss, l_teloss, l_epo = [], [], [], []

        best_loss, best_acc = 0, 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc, precision, recall, f1 = self._test(test_dataloader, criterion)
            l_epo.append(epoch), l_acc.append(test_acc), l_trloss.append(train_loss), l_teloss.append(test_loss)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
                torch.save(self.Mymodel.state_dict(), "bast_model.pth")  # 保存当前较好的模型

            self.logger.info('{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc * 100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.4f}'.format(test_loss, test_acc * 100))
            self.logger.info('[train] r1_tr: {:.4f}, r2_tr: {:.4f},r3_tr: {:.4f}'.format(precision, recall, f1))
            self.logger.info('[test] r1_te: {:.4f}, r2_te: {:.4f},r3_tr: {:.4f}'.format(precision, recall, f1))
        self.logger.info('best loss: {:.4f}, best acc: {:.4f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))

        # Draw the training process
        plt.plot(l_epo, l_acc)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('acc.png')


        plt.plot(l_epo, l_teloss)
        plt.ylabel('test-loss')
        plt.xlabel('epoch')
        plt.savefig('teloss.png')

        plt.plot(l_epo, l_trloss)
        plt.ylabel('train-loss')
        plt.xlabel('epoch')
        plt.savefig('trloss.png')


    def load_test(self):
        self.Mymodel.load_state_dict(torch.load("car_lstm+textcnn2_model"))  # 加载模型
        self.Mymodel.eval()
        y_predict_list = {}
        train_dataloader, test_dataloader = load_dataset(tokenizer=self.tokenizer,
                                                         train_batch_size=self.args.train_batch_size,
                                                         test_batch_size=self.args.test_batch_size,
                                                         model_name=self.args.model_name,
                                                         method_name=self.args.method_name,
                                                         workers=self.args.workers)

        with torch.no_grad():
            iii = 0
            for inputs, targets in tqdm(train_dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                predicts = self.Mymodel(inputs)

                # 保存真实数据和预测数据
                for i in range(len(predicts)):
                    y_predict = predicts[i].detach().numpy().tolist()
                    text = str(train_dataloader.dataset.sentences[iii])
                    positive = str(y_predict[1])
                    iii = iii + 1

                    output = '{}\t{}\n'.format(text, positive)
                    f = codecs.open(r'C:\Users\Chen\Desktop\论文数据\新能源汽车\数据分析\car_sentiment2_2.xlsx', 'a+', 'utf-8')
                    f.write(output)
                    f.close()


# 可以调用其他模块的函数进行运行 from config import get_config
if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    nb = Niubility(args, logger)
    nb.load_test()




