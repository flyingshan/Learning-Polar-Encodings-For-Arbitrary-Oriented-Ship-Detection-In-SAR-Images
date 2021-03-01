import torch
import torch.nn as nn
import os
import numpy as np
import loss
import cv2
import func_utils
from tqdm import tqdm
def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict


class TrainModule(object):
    def __init__(self, dataset, num_classes, model, decoder, down_ratio):
        torch.manual_seed(317)
        self.dataset = dataset
        self.dataset_phase = {'ssdd' : ['train', 'test']}
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.decoder = decoder
        self.down_ratio = down_ratio

    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        }, path)

    def load_model(self, model, optimizer, resume, strict=True):
        """
        给定resume参数(由args中的resume_train给出)，即存档点的文件名信息
        给model加载checkpoint中的模型参数，加载optimizer参数以及epoch数信息
        """
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        # strict参数可以用来选择是否可以容忍加载部分参数（灵活但是容易出问题）
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return model, optimizer, epoch

    def train_network(self, args):
        # 多gpu的情况我就考虑不了了，删掉即可(随时优化无用代码，降低复杂度)
        # if args.ngpus > 1:
        #     if torch.cuda.device_count() > 1:
        #         print("Let's use", torch.cuda.device_count(), "GPUs!")
        #         # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #         self.model = nn.DataParallel(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        # !!!! LR scheduler may not work for low version pytorch.
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)
        save_path = 'weights_' + args.dataset
        start_epoch = 1

        # 当需要加载预训练模型的时候可以指定resume_train参数
        if args.resume_train:
            self.model, self.optimizer, start_epoch = self.load_model(self.model,
                                                                      self.optimizer,
                                                                      args.resume_train,
                                                                      strict=True)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.model.to(self.device)

        # 生成loss.LossAll实例, 用于对检测结果计算loss
        criterion = loss.LossAll()
        print('Setting up data...')

        # self.dataset是一个字典{数据集名：数据集类(在datasets内)}，如果只要一个数据集，考虑优化的简单点
        # 这里利用args参数选择了一个数据集
        dataset_module = self.dataset[args.dataset]

        # 对数据集中每个指定的阶段(在self.dataset_phase中指定)，生成一个类实例（dataset类可以看看实现了啥接口）
        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=self.down_ratio)
                 for x in self.dataset_phase[args.dataset]}

        # 这里作者保留了扩展性，但是其实只用到了一个dsets_loader['train']
        dsets_loader = {}
        # 神奇的dataloader，在run_epoch中可以直接使用 for batch_data in dsets_loader['train']
        #
        dsets_loader['train'] = torch.utils.data.DataLoader(dsets['train'],
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=args.num_workers,
                                                            pin_memory=True,
                                                            drop_last=True,
                                                            collate_fn=collater)

        print('Starting training...')
        train_loss = []
        ap_list = []
        for epoch in range(start_epoch, args.num_epoch + 1):
            print('-' * 10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            # run_epoch方法，封装了前向传播，loss计算，后向传播的代码
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion)
            train_loss.append(epoch_loss)

            # !!! 注意可能造成版本问题的lr scheduler 相关代码
            self.scheduler.step(epoch)

            np.savetxt(os.path.join(save_path, 'train_loss.txt'), train_loss, fmt='%.6f')

            # 隔5保1
            if epoch % 5 == 0: # or epoch > 20:
                self.save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)),
                                epoch,
                                self.model,
                                self.optimizer)

            # 隔5测1（test指的是验证集？）
            if 'test' in self.dataset_phase[args.dataset] and epoch % 5 == 0:
                # 使用dec_eval函数对结果指标进行计算
                mAP = self.dec_eval(args, dsets['test'])
                ap_list.append(mAP)
                np.savetxt(os.path.join(save_path, 'ap_list.txt'), ap_list, fmt='%.6f')

            self.save_model(os.path.join(save_path, 'model_last.pth'),
                            epoch,
                            self.model,
                            self.optimizer)

    def run_epoch(self, phase, data_loader, criterion):
        """封装一个epoch中的forward、loss、backward过程"""
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        # visualize the training process
        data_loader = iter(data_loader)
        for i in tqdm(range(len(data_loader))):
            data_dict = next(data_loader)
            for name in data_dict:
                data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    # 前向传播只调用了model，需要调查一下decoder调用的位置
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)

            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        return epoch_loss

    def dec_eval(self, args, dsets):
        """对检测结果指标进行计算，调用了dataset实例中的dec_evaluation"""
        result_path = 'result_' + args.dataset
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        self.model.eval()
        func_utils.write_results(args,
                                 self.model, dsets,
                                 self.down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path)
        ap = dsets.dec_evaluation(result_path)
        return ap
