"""
Authors: Yi Lin, Wang Jie Xia  
Date: November 10, 2024 - January 10, 2025  
Code adapted from: https://github.com/kjle6/SIF3D-master  
License: [MIT License]  
For inquiries: [17857150773@163.com]
"""

from argparse import ArgumentParser

class MotionFromInentionConfig(ArgumentParser):
    def __init__(self):
        super().__init__()

        self.input_configs = self.add_argument_group('input')
        self.input_configs.add_argument('--batch_size', default=8, type=int)
        self.input_configs.add_argument('--num_workers', default=4, type=int)
        self.input_configs.add_argument('--input_seq_len', default=6, type=int)
        self.input_configs.add_argument('--output_seq_len', default=10, type=int)
        # self.input_configs.add_argument('--dataroot', default='/data/linyi/orion/group/GIMO/', type=str)
        self.input_configs.add_argument('--dataroot', default='/data/wuyang/human/GIMO', type=str)
        self.input_configs.add_argument('--fps', default=2, type=int)
        self.input_configs.add_argument('--sample_points', default=4096, type=int)

        self.motion_configs = self.add_argument_group('motion_prediction')
        self.motion_configs.add_argument('--dropout', default=0.5, type=float)
        self.motion_configs.add_argument('--motion_hidden_dim', default=256, type=int)
        self.motion_configs.add_argument('--N_prox_traj', type=int, default=4, help='Number of prox traj cross-attention repeats')
        self.motion_configs.add_argument('--N_long_traj', type=int, default=4, help='Number of long traj some parameter A')
        
        self.motion_configs.add_argument('--N_k', type=int, default=4, help='Number of prox traj cross-attention repeats')
        self.motion_configs.add_argument('--N_a', type=int, default=4, help='Number of long traj some parameter A')
        self.motion_configs.add_argument('--N_b', type=int, default=8, help='Number of some parameter B')
        self.motion_configs.add_argument('--N_c', type=int, default=8, help='Number of transformer layers for pose')
        self.motion_configs.add_argument('--N_d', type=int, default=8, help='Number of some parameter D')
        self.motion_configs.add_argument('--N_e', type=int, default=8, help='Number of transformer layers for traj')
        self.motion_configs.add_argument('--d_z', type=int, default=128)
        self.motion_configs.add_argument('--end_hidden_dim', type=int, default=256)
        self.motion_configs.add_argument('--motion_label_dim', type=int, default=5, help='Dimension of motion labels')

        self.train_configs = self.add_argument_group('train')
        self.train_configs.add_argument('--save_path', type=str, default='/data/wuyang/MM/output/')
        self.train_configs.add_argument('--save_fre', type=int, default=10)
        self.train_configs.add_argument('--val_fre', type=int, default=1)
        self.train_configs.add_argument('--load_model_dir', type=str, default=None)
        self.train_configs.add_argument('--load_optim_dir', type=str, default=None)
        self.train_configs.add_argument('--train_dimension',type=int, default=128)
        self.train_configs.add_argument('--epoch', type=int, default=200)
        self.train_configs.add_argument('--lr', type=float, default=3e-4)
        self.train_configs.add_argument('--weight_decay', type=float, default=1e-4)
        self.train_configs.add_argument('--gamma', type=float, default=0.98)

        self.eval_configs = self.add_argument_group('eval')
        self.eval_configs.add_argument('--output_path', default='results', type=str)
        self.eval_configs.add_argument('--smplx_path', default='/home/linyi/Code/SIF3D/dataset/', type=str)
        self.eval_configs.add_argument('--vposer_path', default='/home/customer/Yi/vposer_v1_0', type=str)


        self.upSample_configs = self.add_argument_group('upSample')
        self.upSample_configs.add_argument('--nh', type=int, default=1024) #上采样点数
        self.upSample_configs.add_argument('--ng', type=int, default=512)
        self.upSample_configs.add_argument('--humanFeat', type=int, default=16) # encode之后的特征维度
        self.upSample_configs.add_argument('--gazeFeat', type=int, default=32) # encode之后的特征维度
        self.upSample_configs.add_argument('--alpha_traj', type=float, default=0.3) 
        self.upSample_configs.add_argument('--alpha_pose', type=float, default=0.5) 
        self.upSample_configs.add_argument('--gaze_dist_threshold', type=float, default=0.5) 
        
        
        self.upSample_configs.add_argument('--proximity_dim',type=int,default=16)
        self.upSample_configs.add_argument('--long_dim',type=int,default=32)
        self.upSample_configs.add_argument('--use_cluster',type=bool,default=True,help="use cluster for gaze")
        
    def get_configs(self):
        return self.parse_args()


if __name__ == '__main__':
    config = MotionFromInentionConfig()
    print(config.get_configs())