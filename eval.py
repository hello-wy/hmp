"""
Authors: Yi Lin, Wang Jie Xia  
Date: November 10, 2024 - January 10, 2025  
Code adapted from: https://github.com/kjle6/SIF3D-master  
License: [MIT License]  
For inquiries: [17857150773@163.com]
"""

import torch
from torch.utils.data import DataLoader
from dataset import gimo_dataset_with_motion_label
from config.config import MotionFromInentionConfig
from model.model import MotionPred
from utils.logger import MetricTracker


class Motion_evalutor():
    def __init__(self, config):
        self.config = config
        self.test_loader = DataLoader(
            gimo_dataset_with_motion_label.EgoEvalDataset(config, train=False),
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=4,
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def eval(self):
        model = MotionPred(config, ).to(self.device)
        model = model.to(self.device)
        assert self.config.load_model_dir is not None
        print('loading pretrained model from ', self.config.load_model_dir)
        state_dict = torch.load(self.config.load_model_dir)
        state_dict = {k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print('load done!')

        with torch.no_grad():
            model.eval()
            test_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')

            for i, data in enumerate(self.test_loader):
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene, motion_label = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)

                motion_label = motion_label.to(self.device)
                joints_predict, pred_motion_label = model(joints_input[:, :, :23], scene_points, gazes)

                loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
                    self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

                test_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
                test_metrics.update("loss_des_trans", loss_des_trans_gcn, gazes.shape[0])
                test_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
                test_metrics.update("des_mpjpe", des_mpjpe_gcn, gazes.shape[0])

            print(test_metrics.result())
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
    evaluator.eval()