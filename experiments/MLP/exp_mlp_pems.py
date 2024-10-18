import argparse
import time
from datetime import datetime
import numpy as np
import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from experiments.exp_basic import Exp_Basic
from models.MLP.mlp import MLP
from utils.tools import adjust_learning_rate, save_model, load_model
from utils.math_utils import evaluate
import torch.utils.data as torch_data

class ForecastDataset(torch_data.Dataset):
    def __init__(self, df, window_size, horizon, normalize_method=None, norm_statistic=None, interval=1):
        self.window_size = window_size # 12
        self.interval = interval  #1
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        if normalize_method:
            self.data, _ = normalized(self.data, normalize_method, norm_statistic)

    def __getitem__(self, index):
        hi = self.x_end_idx[index] #12
        lo = hi - self.window_size #0
        train_data = self.data[lo: hi] #0:12
        target_data = self.data[hi:hi + self.horizon] #12:24
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx

def normalized(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic

def de_normalized(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data

class Exp_MLP_PEMS(Exp_Basic):
    def __init__(self, args):
        super(Exp_MLP_PEMS, self).__init__(args)
        self.result_file = os.path.join('exp/pems_checkpoint', self.args.dataset, 'checkpoints')

    def _get_data(self):
        data_file = os.path.join('../../data/PEMS', self.args.dataset, self.args.dataset+'.npz')
        print('data file:',data_file)
        data = np.load(data_file,allow_pickle=True)
        data = data['data'][:, :, 0]
        train_ratio = self.args.train_length / (self.args.train_length + self.args.valid_length + self.args.test_length)
        valid_ratio = self.args.valid_length / (self.args.train_length + self.args.valid_length + self.args.test_length)
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        if len(train_data) == 0:
            raise Exception('Cannot organize enough training data')
        if len(valid_data) == 0:
            raise Exception('Cannot organize enough validation data')
        if len(test_data) == 0:
            raise Exception('Cannot organize enough test data')
        if self.args.normtype == 0:
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
            val_mean = np.mean(valid_data, axis=0)
            val_std = np.std(valid_data, axis=0)
            val_normalize_statistic = {"mean": val_mean.tolist(), "std": val_std.tolist()}
            test_mean = np.mean(test_data, axis=0)
            test_std = np.std(test_data, axis=0)
            test_normalize_statistic = {"mean": test_mean.tolist(), "std": test_std.tolist()}
        elif self.args.normtype == 1:
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            train_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
            val_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
            test_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
        else:
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
            val_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
            test_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
        train_set = ForecastDataset(train_data, window_size=self.args.window_size, horizon=self.args.horizon,
                                normalize_method=self.args.norm_method, norm_statistic=train_normalize_statistic)
        valid_set = ForecastDataset(valid_data, window_size=self.args.window_size, horizon=self.args.horizon,
                                    normalize_method=self.args.norm_method, norm_statistic=val_normalize_statistic)
        test_set = ForecastDataset(test_data, window_size=self.args.window_size, horizon=self.args.horizon,
                                    normalize_method=self.args.norm_method, norm_statistic=test_normalize_statistic)
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, drop_last=False, shuffle=True,
                                            num_workers=1)
        valid_loader = DataLoader(valid_set, batch_size=self.args.batch_size, shuffle=False, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False, num_workers=1)
        node_cnt = train_data.shape[1]
        return test_loader, train_loader, valid_loader,node_cnt,test_normalize_statistic,val_normalize_statistic

    def _build_model(self):
        if self.args.dataset == 'PEMS03':
            self.input_dim = 358
        elif self.args.dataset == 'PEMS04':
            self.input_dim = 307
        elif self.args.dataset == 'PEMS07':
            self.input_dim = 883
        elif self.args.dataset == 'PEMS08':
            self.input_dim = 170
        # model = Multi_MLP(node_num=self.input_dim, input_size=self.args.window_size, hidden_sizes=self.args.hidden_sizes, output_size=self.args.horizon)
        model = MLP(input_size=self.args.window_size, hidden_sizes=self.args.hidden_sizes, output_size=self.args.horizon)
        print(model)
        return model

    def _select_optimizer(self):
        if self.args.optimizer == 'RMSProp':
            my_optim = torch.optim.RMSprop(params=self.model.parameters(), lr=self.args.lr, eps=1e-08)
        else:
            my_optim = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999),
                                        weight_decay=self.args.weight_decay)
        return my_optim

    def inference(self, model, dataloader, window_size, horizon):
        forecast_set = []
        target_set = []
        input_set = []
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, target) in enumerate(dataloader):
                inputs = inputs
                target = target
                input_set.append(inputs.detach().cpu().numpy())
                step = 0
                forecast_steps = np.zeros([inputs.size()[0], horizon, inputs.size()[2]], dtype=np.float64)
                # 适配迭代预测和非迭代预测
                while step < horizon:
                    forecast_result = model(inputs)
                    len_model_output = forecast_result.size()[1]
                    if len_model_output == 0:
                        raise Exception('Get blank inference result')
                    inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                    :].clone()
                    inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                    forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                        forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()

                    step += min(horizon - step, len_model_output)
                forecast_set.append(forecast_steps)
                target_set.append(target.detach().cpu().numpy())

        return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0), np.concatenate(input_set,
                                                                                                        axis=0)

    def train(self):
        my_optim = self._select_optimizer()
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=self.args.decay_rate)
        test_loader, train_loader, valid_loader, node_cnt, test_normalize_statistic, val_normalize_statistic = self._get_data()
        forecast_loss = nn.L1Loss()
        best_validate_mae = np.inf
        best_test_mae = np.inf
        validate_score_non_decrease_count = 0

        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, self.result_file, model_name=self.args.dataset,
                                                     horizon=self.args.horizon)
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args.epoch):
            lr = adjust_learning_rate(my_optim, epoch, self.args)
            epoch_start_time = time.time()
            self.model.train()
            loss_total = 0
            cnt = 0
            for i, (inputs, target) in enumerate(train_loader):
                inputs = inputs
                target = target
                self.model.zero_grad()
                forecast = self.model(inputs)
                loss = forecast_loss(forecast, target)
                cnt += 1
                loss.backward()
                my_optim.step()
                loss_total += float(loss)
            print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} '.format(epoch, (
                    time.time() - epoch_start_time), loss_total / cnt))

            if (epoch + 1) % self.args.exponential_decay_step == 0:
                my_lr_scheduler.step()
            if (epoch + 1) % self.args.validate_freq == 0:
                is_best_for_now = False
                print('------ validate on data: VALIDATE ------')
                valid_metrics = self.validate(self.model, valid_loader, self.args.norm_method,
                                                    val_normalize_statistic,
                                                    self.args.window_size, self.args.horizon,
                                                    test=False)
                test_metrics = self.validate(self.model, test_loader, self.args.norm_method,
                                             test_normalize_statistic,
                                             self.args.window_size, self.args.horizon,
                                             test=True)
                if best_validate_mae > valid_metrics['mape']:
                    best_validate_mae = valid_metrics['mape']
                    is_best_for_now = True
                    validate_score_non_decrease_count = 0
                    print('got best validation result:', valid_metrics, test_metrics)
                else:
                    validate_score_non_decrease_count += 1
                if best_test_mae > test_metrics['mape']:
                    best_test_mae = test_metrics['mape']
                    print('got best test result:', test_metrics)

                # save model
                if is_best_for_now:
                    save_model(epoch, lr, model=self.model, model_dir=self.result_file, model_name=self.args.dataset,
                               horizon=self.args.horizon)
                    print('saved model!')
            # early stop
            if self.args.early_stop and validate_score_non_decrease_count >= self.args.early_stop_step:
                break

    def validate(self, model, dataloader, normalize_method, statistic,
                 window_size, horizon, test=False):
        if test:
            print("===================Test Normal=========================")
        else:
            print("===================Validate Normal=========================")
        forecast_norm, target_norm, input_norm = self.inference(model, dataloader, window_size, horizon)
        if normalize_method and statistic:
            forecast = de_normalized(forecast_norm, normalize_method, statistic)
            target = de_normalized(target_norm, normalize_method, statistic)
        else:
            forecast, target, input = forecast_norm, target_norm, input_norm
        score = evaluate(target, forecast)
        score_final_detail = evaluate(target, forecast, by_step=True)
        print('by each step: MAPE & MAE & RMSE', score_final_detail)
        if test:
            print(f'TEST: RAW : MAE {score[1]:7.2f};MAPE {score[0]:7.2f}; RMSE {score[2]:7.2f}.')
        else:
            print(f'VAL: RAW : MAE {score[1]:7.2f};MAPE {score[0]:7.2f}; RMSE {score[2]:7.2f}.')
        return dict(mae=score[1], mape=score[0], rmse=score[2])

    def test(self):
        test_loader, train_loader, valid_loader, node_cnt, test_normalize_statistic, val_normalize_statistic = self._get_data()
        model, lr, epoch = load_model(self.model, self.result_file, model_name=self.args.dataset, horizon=self.args.horizon)
        return self.validate(model, test_loader, self.args.norm_method, test_normalize_statistic,
                 self.args.window_size, self.args.horizon, test=True)

if __name__ == '__main__':

    torch.manual_seed(4321)  # reproducible
    parser = argparse.ArgumentParser(description='MLP on pems datasets')
    ### -------  dataset settings --------------
    parser.add_argument('--dataset', type=str, default='PEMS08',
                        choices=['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'])  # sometimes use: PeMS08
    parser.add_argument('--norm_method', type=str, default='z_score')
    parser.add_argument('--normtype', type=int, default=0)
    ### -------  input/output length settings --------------
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--train_length', type=float, default=6)
    parser.add_argument('--valid_length', type=float, default=2)
    parser.add_argument('--test_length', type=float, default=2)
    ### -------  training settings --------------
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--finetune', type=bool, default=False)
    parser.add_argument('--validate_freq', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--optimizer', type=str, default='N')  #
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--early_stop_step', type=int, default=5)
    parser.add_argument('--exponential_decay_step', type=int, default=5)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--lradj', type=int, default=1, help='adjust learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--model_name', type=str, default='MLP')
    ### -------  model settings --------------
    parser.add_argument('--hidden_sizes', type=list, default=[36])
    args = parser.parse_args()
    Exp=Exp_MLP_PEMS
    exp=Exp(args)

    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        exp.test()
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    elif args.train or args.resume:
        before_train = datetime.now().timestamp()
        print("===================Normal-Start=========================")
        exp.train()
        after_train = datetime.now().timestamp()
        print(f'Training took {(after_train - before_train) / 60} minutes')
        print("===================Normal-End=========================")