import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import numpy as np
from typing import Tuple, List
from model.base_cross_model import CrossAttentionLayer
from model.pointnet_plus2 import PointNet2SemSegSSGShape
from utils.vis_utils import visualize_Scene_wo_color,visualize_SMPLXjoints,visualize_partly_upsample_scene,visualize_gaze
from model.gcn import GCN
from sklearn.cluster import DBSCAN
from pointnet2_ops.pointnet2_utils import ball_query

class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=4, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim * expansion)
        self.fc2 = nn.Linear(out_dim * expansion, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = self.drop(F.gelu(self.fc1(x)))
        out = self.drop(self.fc2(out))
        return out

class PoseEncoder(nn.Module):
    """
    Simple Pose Encoder:
      - (B, T, 23, 3) -> flatten -> MLP -> (B, T, f_p)
    """
    def __init__(self, f_p=128):
        super().__init__()
        self.f_p = f_p
        self.mlp = Mlp(in_dim=69, out_dim=self.f_p, expansion=2, drop=0.1)

    def forward(self, x):
        B, T, J, C = x.shape
        x = x.reshape(B, T, J * C)  # (B, T, 69)
        out = self.mlp(x)          # (B, T, f_p)
        return out

class TrajEncoder(nn.Module):
    """
    Simple Trajectory Encoder:
      - (B, T, 1, 3) -> (B, T, 3) -> MLP -> (B, T, f_r)
    """
    def __init__(self, f_r=64):
        super().__init__()
        self.f_r = f_r
        self.mlp = Mlp(in_dim=3, out_dim=self.f_r, expansion=2, drop=0.1)

    def forward(self, traj):
        B, T, _, C = traj.shape
        traj = traj.reshape(B, T, C)  # (B, T, 3)
        out = self.mlp(traj)         # (B, T, f_r)
        return out

class TransformerFilling(nn.Module):
    """
    A Transformer used for filling predictions in sequences with masks.
    This is demonstrated using PyTorch's built-in nn.TransformerEncoder/Decoder.
    """
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (T, B, D)  # Source sequence with shape (T, B, D)
        tgt: (T, B, D)  # Target sequence with shape (T, B, D)
        src_mask: Mask for the source sequence (optional)
        tgt_mask: Mask for the target sequence (optional)
        """
        memory = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        out = self.transformer_decoder(tgt, memory, tgt_mask, src_mask)
        return out

class EndingMotionGenerator(nn.Module):
    """
    Maps the learned motion type embedding z (dimension d_z)
    to a 23×3 = 69-dimensional ending pose (or first maps to 23, then maps to 23×3).
    It can also directly map to (23×3) dimensions.
    """

    def __init__(self, d_in=64, d_hidden=128, d_out=69):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out),  # Directly output 69-dimensional -> (23×3)
        )

    def forward(self, z):
        """
        z: (B, d_in)
        return: ending_motion_3d (B, 23, 3)
        """
        B, _ = z.shape
        out = self.mlp(z)               # (B, d_out=69)
        out = out.view(B, 23, 3)        # reshape成 (B, 23, 3)
        return out


class SceneFeatureExtractor(nn.Module):
    def __init__(self,config, npoints_init: int):
        super(SceneFeatureExtractor, self).__init__()
        self.npoints_init = npoints_init
        self.dimension = config.train_dimension
        
        self.encoder1 = PointNet2SemSegSSGShape({'feat_dim': 16,'input_dim':3 })
        self.encoder2 = PointNet2SemSegSSGShape({'feat_dim': 32,'input_dim':13})
        self.encoder3 = PointNet2SemSegSSGShape({'feat_dim': 64,'input_dim':29})
        self.encoder4 = PointNet2SemSegSSGShape({'feat_dim': 128,'input_dim':61})
        
    def forward(self, scene_points: torch.Tensor) -> List[torch.Tensor]:
        B, N, _ = scene_points.shape

        def downSample(points, npoints):
            # 使用 FPS 下采样，直接在 GPU 上操作
            batch_size, _, _ = points.shape
            new_idx = furthest_point_sample(points[:,:,:3].contiguous(), npoints)  # [B, npoints]
            sampled_points = points[torch.arange(batch_size)[:, None], new_idx]  # [B, npoints, 3]
            
            return sampled_points
        
        # Step 1: Downsample to N/2 and encode
        scene_down1 = downSample(scene_points, N // 2)  # [B, N/2, 3]
        scene_feat1,_ = self.encoder1(scene_down1.repeat(1,1,2))  # [B, N/2, 16]
        scene_feat1 = scene_feat1.transpose(1, 2) 
        scene_feat1 = torch.cat([scene_down1[:, :, :3], scene_feat1[:, :, 3:]], dim=-1) # 替换前三维

        
        # Step 2: Downsample to N/4 and encode
        scene_down2 = downSample(scene_feat1, N // 4)  # [B, N/4, 16]
        scene_feat2,_ = self.encoder2(scene_down2)  # [B, N/4, 32]
        scene_feat2 = scene_feat2.transpose(1, 2) 
        scene_feat2 = torch.cat([scene_down2[:, :, :3], scene_feat2[:, :, 3:]], dim=-1) # 替换前三维

        # Step 3: Downsample to N/8 and encode
        scene_down3 = downSample(scene_feat2, N // 8)  # [B, N/8, 32]
        scene_feat3,_ = self.encoder3(scene_down3)  # [B, N/8, 64]
        scene_feat3 = scene_feat3.transpose(1, 2) 
        scene_feat3 = torch.cat([scene_down3[:, :, :3], scene_feat3[:, :, 3:]], dim=-1) # 替换前三维

        # Step 4: Downsample to N/16 and encode
        scene_down4 = downSample(scene_feat3, N // 16)  # [B, N/16, 64]
        scene_feat4,_ = self.encoder4(scene_down4)  # [B, N/16, 128]
        scene_feat4 = scene_feat4.transpose(1, 2) 
        scene_feat4 = torch.cat([scene_down4[:, :, :3], scene_feat4[:, :, 3:]], dim=-1) # 替换前三维
        
        # visualize_Scene_wo_color(scene_feat1[0,:,:3].cpu().detach().numpy(), "/data/wuyang/2048")
        # visualize_Scene_wo_color(scene_feat2[0,:,:3].cpu().detach().numpy(), "/data/wuyang/1024")
        # visualize_Scene_wo_color(scene_feat3[0,:,:3].cpu().detach().numpy(), "/data/wuyang/512")
        # visualize_Scene_wo_color(scene_feat4[0,:,:3].cpu().detach().numpy(), "/data/wuyang/256")
        return scene_feat1, scene_feat2, scene_feat3, scene_feat4


class HumanGazeSceneUpSample(nn.Module):
    def __init__(self, config):
        super(HumanGazeSceneUpSample, self).__init__()
        self.nh = config.nh  # Number of points for human-centered
        self.ng = config.ng  # Number of points for gaze-centered
        self.use_cluster = config.use_cluster
        
        self.humanEncoder = PointNet2SemSegSSGShape({'feat_dim': config.humanFeat})
        self.gazeEncoder = PointNet2SemSegSSGShape({'feat_dim': config.gazeFeat})
    
    def cutScene(self,scene_data, pelvis_seq, num_points):
        """
        scene_data:场景数据 (N,3)
        pelvis_seq: pelvis坐标 (len,3)
        num_points: 采样点数 
        """
        TRAJ_PAD = 2.0
        REGION_SIZE = 2.0

        scene_trans = np.array([0,0,0],dtype=np.float32).reshape(1,3) #TODO: 1,3 scene_trans ??

        traj_max = pelvis_seq.max(axis=0)[0:2] # [2,] 最大点（x，y）
        traj_min = pelvis_seq.min(axis=0)[0:2] # [2,] 最小的
        traj_size = traj_max - traj_min # [2,] 
        traj_size = traj_size + TRAJ_PAD * np.exp(-traj_size)   # 轨迹变大

        pad = (REGION_SIZE - traj_size) / 2
        pad = np.maximum(pad, [0, 0])

        center = (traj_max + traj_min) / 2 # [2,] 中间点通过计算平均值
        center_region_max = center + pad
        center_region_min = center - pad
        sample_xy = (center_region_min + center_region_max)/2 # 采样中心点
        sample_region_max = sample_xy + REGION_SIZE / 2
        sample_region_min = sample_xy - REGION_SIZE / 2

        scene_pcd = scene_data.copy() # 8192,3
        scene_pcd[:, 0:3] += scene_trans  
        point_in_region = (scene_pcd[:, 0] >= sample_region_min[0]) & (scene_pcd[:, 0] <= sample_region_max[0]) & \
                            (scene_pcd[:, 1] >= sample_region_min[1]) & (scene_pcd[:, 1] <= sample_region_max[1])
        
        
        indices = np.arange(len(scene_pcd))[point_in_region] # point_in_region 中点的索引  3271
        assert len(indices) > 0, "No points in the region!"
        if len(indices) < num_points:
            if len(indices) < num_points // 4:
                print(f"Warning: only {len(indices)} points in the region! Less than {num_points // 4} points!")
            while len(indices) < num_points:
                indices = np.concatenate([indices, indices])    
        indices = np.random.choice(indices, num_points, replace=False)

        points = scene_data.copy()
        points[:, 0:3] += scene_trans
        points = points[indices]
            
        return points

    def batchUpSample(self,scene_data,joints,gazes):
        human_scene=[]
        gaze_scene=[]
        for i in range(scene_data.shape[0]):
            scene = scene_data[i].cpu().numpy()    
            
            pose_seq = joints[i].cpu().numpy()
            pose_seq = pose_seq.copy().astype(np.float32)
            pelvis_seq = pose_seq[:,[0],:].reshape(-1,3) # len, 3 所有帧的pelvis坐标
            scene_points = self.cutScene(scene, pelvis_seq, self.nh)
                
            gaze = gazes[i].cpu().numpy()
            if self.use_cluster:
                db = DBSCAN(eps=0.5, min_samples=1).fit(gaze.reshape(-1, 3))
                labels = db.labels_
                largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
                close_indices = np.where(labels == largest_cluster)[0]
                gaze = gaze[close_indices][:,0,:]
            else:
                gaze = gaze[0:,0,:]    # 第一帧 
                
            gaze_points = self.cutScene(scene, gaze, self.ng)
            
            human_scene.append(scene_points)
            gaze_scene.append(gaze_points)
        
        human_scene = np.array(human_scene)  
        human_scene = torch.from_numpy(human_scene).cuda()
        
        gaze_scene = np.array(gaze_scene)  
        gaze_scene = torch.from_numpy(gaze_scene).cuda()
        
        return human_scene,gaze_scene
    
    def humanUpSample(self,scene_data,joints):
        human_scene=[]
        for i in range(scene_data.shape[0]):
            scene = scene_data[i].cpu().numpy()    
            
            pose_seq = joints[i].cpu().numpy()
            pose_seq = pose_seq.copy().astype(np.float32)
            pelvis_seq = pose_seq[:,[0],:].reshape(-1,3) # len, 3 所有帧的pelvis坐标
            scene_points = self.cutScene(scene, pelvis_seq, self.nh)
            
            human_scene.append(scene_points)
        human_scene = np.array(human_scene)  
        human_scene = torch.from_numpy(human_scene).cuda()
        return human_scene
    
    def forward(self, scene_points: torch.Tensor, joints: torch.Tensor, 
                gazes: torch.Tensor, ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Human-centered upsampling
        human_center_points,gaze_center_points = self.batchUpSample(
            scene_points, joints, gazes
        )
        
        # visualize_partly_upsample_scene(human_center_points[0,:,:3].cpu().detach().numpy(),"/data/wuyang/MM/upSample/humancenter")
        # visualize_partly_upsample_scene(gaze_center_points[0,:,:3].cpu().detach().numpy(), "/data/wuyang/MM/upSample/gazecenter")
        # visualize_SMPLXjoints(joints[0,:,:,:].cpu(),scene_points[0,:,:].cpu(),0,"/data/wuyang/MM/upSample")
        # visualize_gaze(gazes[0,:,0,:].cpu(),0,"/data/wuyang/MM/upSample")
        
        human_center_feat,_ = self.humanEncoder(human_center_points.repeat(1,1,2))  # [B, Nh, 16]
        human_center_feat = human_center_feat.transpose(1, 2)
        
        gaze_center_feat,_ = self.gazeEncoder(gaze_center_points.repeat(1,1,2))# [B, Ng, 32]
        gaze_center_feat = gaze_center_feat.transpose(1, 2)
        
        return human_center_feat, gaze_center_feat


class TrajectoryPlanner(nn.Module):
    def __init__(self,config, dim: int = 64):
        super(TrajectoryPlanner, self).__init__()
        
        self.dimension = dim
        self.traj_encoder = TrajEncoder(f_r = dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = config.alpha_traj
        self.seq_len = config.input_seq_len + config.output_seq_len
        
        self.proximity_traj_cross_attn = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.dimension, num_kv_channels=config.proximity_dim, num_heads=1)
            for _ in range(config.N_prox_traj)
        ])
        
        self.long_traj_cross_attn = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.dimension, num_kv_channels=config.long_dim, num_heads=1)
            for _ in range(config.N_long_traj)
        ])
        
        self.traj_input_mlp   = nn.Linear(self.dimension, self.dimension, bias=False).to(self.device)
        self.traj_transformer = TransformerFilling(d_model=self.dimension,  nhead=8, num_layers=config.N_e)
        self.traj_output_mlp  = nn.Linear(self.dimension, 3, bias=False).to(self.device)


    def forward(self, scene_feat3,human_scene_feat,gaze_scene_feat,gazes):
        """
        Plan next trajectory using transformer.
        """
        B = gazes.shape[0]
        gaze_feat = self.traj_encoder(gazes)  # [8, 6, 128]
        
        gazes_label = gazes[:,[-1],:,:].repeat(1,10,1,1)
        gazes_filled = torch.cat([gazes,gazes_label],dim = 1)
        gaze_feat_filled = self.traj_encoder(gazes_filled)  # [8, 6, 128]
        
        traj_feat_proximity = gaze_feat_filled.clone() # [8, 16, 128] 近距离 traj
        for layer in self.proximity_traj_cross_attn:
            traj_feat_proximity = layer(traj_feat_proximity, human_scene_feat)
        
        traj_feat_long = gaze_feat_filled.clone()  # 远距离 traj
        for layer in self.long_traj_cross_attn:
            traj_feat_long = layer(traj_feat_long, gaze_scene_feat)

        traj_feat_fused = self.alpha * traj_feat_proximity + (1 - self.alpha) * traj_feat_long  #8,16,64
        traj_feat = torch.cat([gaze_feat, traj_feat_fused[:,6:,:]],dim = 1)
        traj_feat = self.traj_input_mlp(traj_feat)
        
        H = scene_feat3[..., [2]]
        phi,_ = torch.max(H, dim=1, keepdim=True)
        phi /= 5.0
        M = (H > phi).float()
        scene_walkable_feat = scene_feat3 * (1 - M)  # (B, 512, 64) lower then phi can walk
        
        pred_traj_embed = self.traj_transformer(scene_walkable_feat.transpose(0, 1),traj_feat.transpose(0, 1)) 
        
        pred_traj = self.traj_output_mlp(pred_traj_embed.transpose(0,1))  # (B, T, 3)
        
        return pred_traj.reshape(B, self.seq_len, 1, 3)

class PoseReferencer(nn.Module):
    def __init__(self,config, dim: int = 128, transformer_dim: int = 128):
        super(PoseReferencer, self).__init__()
        
        self.dimension = dim
        self.alpha = config.alpha_pose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.joint_num = 23
        self.input_seq_len  = config.input_seq_len  # T
        self.output_seq_len = config.output_seq_len # ΔT
        self.seq_len = self.input_seq_len + self.output_seq_len
        
        self.pose_encoder = PoseEncoder(f_p=self.dimension)
        self.proximity_pose_cross_attn = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.dimension, num_kv_channels=config.proximity_dim, num_heads=1)
            for _ in range(config.N_b)
        ])
        self.long_pose_cross_attn = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.dimension, num_kv_channels=config.long_dim, num_heads=1)
            for _ in range(config.N_b)
        ])
        
        self.label_to_z = nn.Linear(config.motion_label_dim, config.d_z)  # (B, d_label) -> (B, d_z)
        self.motion_type_mlp = Mlp(in_dim=self.dimension + self.dimension, out_dim=config.motion_label_dim, expansion=2, drop=0.1)
        
        self.ending_gen = EndingMotionGenerator(
            d_in=config.d_z,
            d_hidden=config.end_hidden_dim,
            d_out=69,  # 23×3
        )
        
        self.pose_input_proj  = nn.Linear(69, self.dimension, bias=False).to(self.device)
        self.pose_output_proj = nn.Linear(self.dimension, 69, bias=False).to(self.device)
        self.pose_transformer = TransformerFilling(d_model=self.dimension, nhead=8, num_layers=config.N_c)
        
        self.pose_transformer = TransformerFilling(d_model=self.dimension, nhead=8, num_layers=config.N_e)

    def forward(self, scene_feat3,human_scene_feat,gaze_scene_feat, human_pose) -> torch.Tensor:
        pose_feat = self.pose_encoder(human_pose) 
    
        pose_feat_proximity = pose_feat.clone() # [8, 16, 128] 近距离 traj
        for layer in self.proximity_pose_cross_attn:
            pose_feat_proximity = layer(pose_feat_proximity, human_scene_feat)
        
        pose_feat_long = pose_feat.clone()  # 远距离 traj
        for layer in self.long_pose_cross_attn:
            pose_feat_long = layer(pose_feat_long, gaze_scene_feat)
        
        pose_feat_fused = self.alpha * pose_feat_proximity + (1 - self.alpha) * pose_feat_long  #8,16,128

        # pred_pose_embed = self.pose_transformer(pose_feat_fused.transpose(0, 1),
        #                                         pose_feat_fused.transpose(0, 1)) 
        
        # 辅助任务动作类型识别 （远距离感知情况）
        cat_feat = torch.cat([pose_feat_long, pose_feat], dim=-1)
        motion_logits = self.motion_type_mlp(cat_feat)
        motion_logits_last = motion_logits[:, -1, :]
        motion_label = torch.argmax(motion_logits_last, dim=-1)
        
        z = self.label_to_z(motion_logits_last)  # [8, 128]
        ending_motion = self.ending_gen(z) # 最后一帧的motion
        
        B, T, J, C = human_pose.shape
        # future_mask_joints = torch.zeros((B, self.output_seq_len, J, C), device=human_pose.device)
        future_mask_joints = human_pose[:,[T-1],:,:].repeat(1,10,1,1)
        motion_masked = torch.cat([human_pose, future_mask_joints], dim=1)  # (B, T+ΔT, 23, 3)
        
        motion_masked[:, -1, :, :] = ending_motion[:]
        motion_masked_flat = motion_masked.reshape(B, self.seq_len, -1) # [8, 16, 69] 
        motion_embed = self.pose_input_proj(motion_masked_flat).transpose(0, 1) # [16, 8, 128]

        pred_embed = self.pose_transformer(motion_embed, motion_embed)  # (T, B, 128)
        pred_embed = pred_embed.transpose(0, 1)               # (B, T, 128)
        pred_flat  = self.pose_output_proj(pred_embed)        # (B, T, 69)
        pred_motion = pred_flat.reshape(B, self.seq_len, self.joint_num, 3)
        
        return  pose_feat_fused, pred_motion, motion_label

class Predictor(nn.Module):
    def __init__(self,config, dim: int = 128):
        super(Predictor, self).__init__()
        self.dimension = dim
        self.seq_len = config.input_seq_len + config.output_seq_len
        self.motion_GCN = GCN(config, node_n=69)
        self.motion_GCN2 = GCN(config, node_n=69)


    def forward(self,human_pose,pred_motion,pred_traj) -> torch.Tensor:
        """
        Reference next pose using transformer.
        """
        B = human_pose.shape[0]
        pred_motion[:, :6, :, :] = human_pose
        distance = pred_motion[:,:,[0],:] - pred_traj.reshape(B, self.seq_len, 1, 3)
        pred_motion =  pred_motion - distance
        pred_motion[:, :, [0], :] = pred_traj  # (B, T+ΔT, 23, 3)

        for _ in range(1):
            pred_motion = self.motion_GCN(pred_motion)
            pred_motion[:, :6, :, :] = human_pose
            pred_motion = self.motion_GCN2(pred_motion)
        return pred_motion


class FullModel(nn.Module):
    def __init__(self,config, npoints_init: int = 4096):
        super(FullModel, self).__init__()
        self.dimension = config.train_dimension
                # 1) Pose Encoder
        self.pose_encoder = PoseEncoder(f_p=self.dimension)

        # 2) Scene Encoder
        self.scene_encoder = PointNet2SemSegSSGShape({'feat_dim': self.dimension})

        # 3) Trajectory Encoder
        self.traj_encoder = TrajEncoder(f_r=self.dimension)
        self.ending_gen = EndingMotionGenerator(
            d_in=config.d_z,
            d_hidden=config.end_hidden_dim,
            d_out=69,  # 23×3
        )
        
        self.scene_extractor = SceneFeatureExtractor(config,npoints_init)
        self.human_gaze_upsample = HumanGazeSceneUpSample(config)
        self.trajectory_planner = TrajectoryPlanner(config=config)
        self.pose_referencer = PoseReferencer(config=config)
        self.predictor = Predictor(config=config)

        
        
    def forward(self, scene_points: torch.Tensor, human_pose: torch.Tensor, gazes: torch.Tensor) -> Tuple:
        """
        human_pose: [B,6,23,3]
        gazes:[B,6,1,3]
        """
        # Global scene feature extraction
        scene_feat1, scene_feat2, scene_feat3, scene_feat4 = self.scene_extractor(scene_points)
        
        # Human/Gaze-centered scene upsampling feat
        human_center_points, gaze_center_points = self.human_gaze_upsample(scene_points, human_pose, gazes)

        human_scene_feat = torch.cat([scene_feat1,human_center_points],dim = 1) # [B,N/2 + Nh ,16]
        gaze_scene_feat = torch.cat([scene_feat2,gaze_center_points],dim = 1)
        
        pred_traj = self.trajectory_planner(scene_feat3,human_scene_feat,gaze_scene_feat,gazes)
        pose_feat_fused, pred_motion, motion_label = self.pose_referencer(scene_feat3,human_scene_feat,gaze_scene_feat,human_pose)
        
        pred_motion = self.predictor(human_pose,pred_motion,pred_traj)
        
        return pred_motion, motion_label
