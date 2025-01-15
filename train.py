"""
Authors: Yi Lin, Wang Jie Xia  
Date: November 10, 2024 - January 10, 2025  
Code adapted from: https://github.com/kjle6/SIF3D-master  
License: [MIT License]  
For inquiries: [17857150773@163.com]
"""

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from dataset import gimo_dataset_with_motion_label
from config.config import MotionFromInentionConfig
from model.model import MotionPred
from utils.logger import create_logger, MetricTracker


class Motion_evalutor():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = MotionPred(config, ).to(self.device)
        if self.config.load_model_dir is not None:
            state_dict = torch.load(self.config.load_model_dir)
            state_dict = {k: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print('load done!')

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_adjuster = ExponentialLR(self.optim, gamma=config.gamma)

        self.train_dataset = gimo_dataset_with_motion_label.EgoEvalDataset(config, train=True)
        self.test_dataset = gimo_dataset_with_motion_label.EgoEvalDataset(config, train=False)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=9, # do not change
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
        self.label_loss_fn = nn.CrossEntropyLoss()
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path, exist_ok=True)
        self.logger = create_logger(config.save_path)

    def train(self):
        train_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')

        for epoch in range(config.epoch):
            for data in tqdm(self.train_loader):
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene, motion_label = data

                gazes = gazes.to(self.device) 
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device) 
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)
                motion_label = motion_label.to(self.device)

                joints_predict, pred_motion_label = self.model(joints_input[:, :, :23], scene_points, gazes)
                loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])
                
                B = joints_label.shape[0]
                predict_label = torch.zeros(B, 5).cuda()
                for i in range(B):
                    predict_motion = torch.zeros(5).cuda()
                    predict_motion[pred_motion_label[i]] = 1
                    predict_label[i] = predict_motion
                loss_label = self.label_loss_fn(predict_label.float(), motion_label)
                
                loss = loss_trans_gcn.mean() + loss_des_trans_gcn + mpjpe_gcn.mean() + des_mpjpe_gcn + loss_label
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
                train_metrics.update("loss_des_trans", loss_des_trans_gcn.mean(), gazes.shape[0])
                train_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
                train_metrics.update("des_mpjpe", des_mpjpe_gcn, gazes.shape[0])

            self.lr_adjuster.step()
            train_metrics.log(self.logger, epoch,train=True)
            train_metrics.reset()

            if epoch % config.val_fre == 0:
                self.model.eval()
                with torch.no_grad():
                    self.test(epoch)
                self.model.train()
            if epoch % config.save_fre == 0:
                torch.save(self.model.state_dict(), f"{self.config.save_path}/{epoch}.pth")

    def test(self, epoch):
        test_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')

        for i, data in enumerate(self.test_loader):
            gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene ,motion_label= data

            gazes = gazes.to(self.device)
            poses_input = poses_input.to(self.device)
            poses_label = poses_label.to(self.device)
            scene_points = scene_points.to(self.device).contiguous()
            joints_input = joints_input.to(self.device)
            joints_label = joints_label.to(self.device)

            joints_predict,motion_label = self.model(joints_input[:, :, :23], scene_points, gazes)
            loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])
            
            test_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
            test_metrics.update("loss_des_trans", loss_des_trans_gcn, gazes.shape[0])
            test_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
            test_metrics.update("des_mpjpe", des_mpjpe_gcn, gazes.shape[0])

        test_metrics.log(self.logger, epoch, train=False)
        test_metrics.reset()

    def calc_loss_gcn(self, poses_predict, poses_label, poses_input):
        poses_label = torch.cat([poses_input, poses_label], dim=1)
        loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)
        poses_label = poses_label - poses_label[:, :, [0]]
        poses_predict = poses_predict - poses_predict[:, :, [0]]
        mpjpe = torch.norm(poses_predict - poses_label, dim=-1)
        return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()


if __name__ == '__main__':
    config = MotionFromInentionConfig().parse_args()
    evaluator = Motion_evalutor(config)
    evaluator.train()
