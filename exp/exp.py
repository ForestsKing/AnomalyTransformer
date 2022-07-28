import os
from time import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import myDataset
from model.AnomalyTransformer import AnomalyTransformer
from utils.adjustpredicts import adjust_predicts
from utils.earlystoping import EarlyStopping
from utils.evalmethods import bestf1_threshold
from utils.getdata import get_data


class Exp:

    def __init__(self, config):
        self.__dict__.update(config)
        self._get_data()
        self._get_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_data(self):
        data = get_data()
        if self.verbose:
            for k, v in data.items():
                print(k, ': ', v.shape)

        trainset = myDataset(data['train_data'], data['train_label'], self.windows_size, 1)
        validset = myDataset(data['valid_data'], data['valid_label'], self.windows_size, 1)
        threset = myDataset(data['thre_data'], data['thre_label'], self.windows_size, self.windows_size)
        testset = myDataset(data['test_data'], data['test_label'], self.windows_size, self.windows_size)

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=False)
        self.threloader = DataLoader(threset, batch_size=self.batch_size, shuffle=False)
        self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

    def _get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AnomalyTransformer().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.5 ** ((epoch - 1) // 2))
        self.early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose,
                                            path=self.model_dir + 'model.pkl')
        self.MSE = nn.MSELoss(reduction='none')
        self.KL = nn.KLDivLoss(reduction='none')
        print(self.device)

    def _get_AssociationDiscrepancy(self, series, prior):
        # P: [batch_size, layer_num, windows_size, windows_size]
        # S: [batch_size, layer_num, windows_size, windows_size]
        P = torch.mean(prior, dim=2)
        S = torch.mean(series, dim=2)

        # R: [batch_size, layer_num, windows_size]
        R = torch.sum(self.KL(P, S), dim=-1) + torch.sum(self.KL(S, P), dim=-1)

        # R: [batch_size, windows_size]
        R = torch.mean(R, dim=1)
        return R

    def _get_loss(self, batch_x, output, series, prior):
        # 将 prior 除以行和归一化，否则 KL 散度可能为负值
        prior = prior / torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1)

        # detach()创建一个新的 Tensor，仍指向原变量的存放位置，但requires_grad为false
        series_loss = torch.mean(torch.abs(self._get_AssociationDiscrepancy(series, prior.detach())))
        prior_loss = torch.mean(torch.abs(self._get_AssociationDiscrepancy(series.detach(), prior)))
        rec_loss = torch.mean(torch.mean(self.MSE(output, batch_x), dim=-1))
        return series_loss, prior_loss, rec_loss

    def _get_cri(self, batch_x, output, series, prior):
        metric = torch.softmax(-self._get_AssociationDiscrepancy(series, prior) * self.temperature, dim=-1)
        loss = torch.mean(self.MSE(output, batch_x), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        return cri

    def train(self):
        for e in range(self.epochs):
            start = time()

            self.model.train()
            train_loss_1, train_loss_2 = [], []

            for (batch_x, _) in tqdm(self.trainloader):
                # batch_x: [batch_size, windows_size, feature_dim]
                # batch_y: [batch_size, windows_size]
                batch_x = batch_x.float().to(self.device)
                output, series, prior = self.model(batch_x)

                # output: [batch_size, windows_size, feature_dim]
                # series: [batch_size, layer_num, head_num, windows_size, windows_size]
                # prior: [batch_size, layer_num, head_num, windows_size, windows_size]
                series_loss, prior_loss, rec_loss = self._get_loss(batch_x, output, series, prior)

                loss1 = rec_loss + self.k * prior_loss
                loss1.backward(retain_graph=True)  # retain_graph可以再次backward
                self.optimizer.step()
                self.optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                output, series, prior = self.model(batch_x)
                series_loss, prior_loss, rec_loss = self._get_loss(batch_x, output, series, prior)

                loss2 = rec_loss - self.k * series_loss
                loss2.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                train_loss_1.append(loss1.item())
                train_loss_2.append(loss2.item())

            with torch.no_grad():
                self.model.eval()
                valid_loss_1, valid_loss_2 = [], []

                for (batch_x, _) in tqdm(self.validloader):
                    batch_x = batch_x.float().to(self.device)
                    output, series, prior = self.model(batch_x)
                    series_loss, prior_loss, rec_loss = self._get_loss(batch_x, output, series, prior)

                    loss1 = rec_loss + self.k * prior_loss
                    loss2 = rec_loss - self.k * series_loss

                    valid_loss_1.append(loss1.item())
                    valid_loss_2.append(loss2.item())

            train_loss_1, train_loss_2 = np.average(train_loss_1), np.average(train_loss_2)
            valid_loss_1, valid_loss_2 = np.average(valid_loss_1), np.average(valid_loss_2)

            end = time()
            print("Epoch: {0} || Train Loss 1: {1:.4f} | Train Loss 2: {2:.4f} || "
                  "Vali Loss 1: {3:.4f} | Vali Loss 2: {4:.4f} || Cost: {5:.4f} s".format(
                e + 1, train_loss_1, train_loss_2, valid_loss_1, valid_loss_2, end - start))

            self.early_stopping(valid_loss_1, valid_loss_2, self.model)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()

        self.model.load_state_dict(torch.load(self.model_dir + 'model.pkl'))

    def test(self):
        self.model.load_state_dict(torch.load(self.model_dir + 'model.pkl'))

        with torch.no_grad():
            self.model.eval()

            thre_score, thre_label = [], []
            for (batch_x, batch_y) in tqdm(self.threloader):
                batch_x = batch_x.float().to(self.device)
                output, series, prior = self.model(batch_x)
                score = self._get_cri(batch_x, output, series, prior)
                label = batch_y.detach().cpu().numpy()
                thre_score.append(score)
                thre_label.append(label)

            test_score, test_label = [], []
            for (batch_x, batch_y) in tqdm(self.testloader):
                batch_x = batch_x.float().to(self.device)
                output, series, prior = self.model(batch_x)
                score = self._get_cri(batch_x, output, series, prior)
                label = batch_y.detach().cpu().numpy()
                test_score.append(score)
                test_label.append(label)

        thre_score = np.concatenate(thre_score, axis=0).reshape(-1)
        thre_label = np.concatenate(thre_label, axis=0).reshape(-1)
        thresh = bestf1_threshold(thre_score, thre_label, self.adjust)
        print("Threshold :", thresh)

        test_score = np.concatenate(test_score, axis=0).reshape(-1)
        test_label = np.concatenate(test_label, axis=0).reshape(-1)
        test_pred = (test_score > thresh).astype(int)

        if self.adjust:
            test_pred = adjust_predicts(test_label, test_pred)

        accuracy = accuracy_score(test_label, test_pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(test_label, test_pred, average='binary')
        print("Accuracy : {0:.4f} | Precision : {1:.4f} | Recall : {2:.4f} | F-score : {3:.4f} ".format(
            accuracy, precision, recall, f_score))
