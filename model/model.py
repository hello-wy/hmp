"""
Authors: Yi Lin, Wang Jie Xia  
Date: November 10, 2024 - January 10, 2025  
Code adapted from: https://github.com/kjle6/SIF3D-master  
License: [MIT License]  
For inquiries: [17857150773@163.com]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_cross_model import CrossAttentionLayer
from model.pointnet_plus2 import PointNet2SemSegSSGShape
import numpy as np
from model.gcn import GCN
import torch.nn.init as init


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


class MotionPred(nn.Module):
    """
    Core Model:
     - Integrates PoseEncoder / SceneEncoder / TrajEncoder
     - Multimodal cross-attention
     - Uses differentiable priors for "ending motion" generation
     - Uses TransformerFilling to complete sequence padding
     - Trajectory prediction remains similar to the original scheme
    """
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.input_seq_len  = config.input_seq_len  # T
        self.output_seq_len = config.output_seq_len # ΔT
        self.seq_len = self.input_seq_len + self.output_seq_len
        self.joint_num = 23
        self.dimension = config.train_dimension
        
        # ========== Module Definitions ==========

        # 1) Pose Encoder
        self.pose_encoder = PoseEncoder(f_p=self.dimension)

        # 2) Scene Encoder
        self.scene_encoder = PointNet2SemSegSSGShape({'feat_dim': self.dimension})

        # 3) Trajectory Encoder
        self.traj_encoder = TrajEncoder(f_r=self.dimension)

        # 4) Multimodal cross-attention
        self.f_s_cross_att = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.dimension, num_kv_channels=1, num_heads=1)
            for _ in range(config.N_k)
        ])
        self.xs_cross_att = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.dimension, num_kv_channels=self.dimension, num_heads=1)
            for _ in range(config.N_a)
        ])
        self.ts_cross_att = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.dimension, num_kv_channels=self.dimension, num_heads=1)
            for _ in range(config.N_b)
        ])
        self.fm_cross_att = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.dimension, num_kv_channels=self.dimension, num_heads=1)
            for _ in range(config.N_b)
        ])
        
        # 5) Motion Type Output & Embedding
        #    First output motion_label_dim size, then map to differentiable prior embedding z
        self.motion_type_mlp = Mlp(in_dim=self.dimension + self.dimension, out_dim=config.motion_label_dim, expansion=2, drop=0.1)
        self.label_to_z = nn.Linear(config.motion_label_dim, config.d_z)  # (B, d_label) -> (B, d_z)

        # 6) Differentiable Prior: Ending Motion Generator
        self.ending_gen = EndingMotionGenerator(
            d_in=config.d_z,
            d_hidden=config.end_hidden_dim,
            d_out=69,  # 23×3
        )

        # 7) Pose Decoder and Traj Decoder
        self.pose_transformer = TransformerFilling(d_model=self.dimension, nhead=8, num_layers=config.N_c)
        self.traj_transformer = TransformerFilling(d_model=self.dimension,  nhead=8, num_layers=config.N_e)

        # 8) General input/output projections
        self.pose_input_proj  = nn.Linear(69, self.dimension, bias=False).to(self.device)
        self.pose_output_proj = nn.Linear(self.dimension, 69, bias=False).to(self.device)

        self.traj_input_mlp   = nn.Linear(self.dimension, self.dimension, bias=False).to(self.device)
        self.traj_output_mlp  = nn.Linear(self.dimension, 3, bias=False).to(self.device)

        # 9) Optional: Trajectory Post-processing / Motion Post-processing
        #    Example of a GCN or other post-processing network
        self.motion_GCN = GCN(config, node_n=69)
        self.motion_GCN2 = GCN(config, node_n=69)
        
    def _initialize_weights(self):
        """
        Initialize all parameters in the model, including Conv2d, Linear, LSTM, Transformer, etc.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if isinstance(param, nn.Parameter):
                    # For nn.Parameter type
                    init.normal_(param, mean=0.0, std=0.02)  # General normal distribution initialization
                elif isinstance(param, nn.Conv2d):
                    # Use He initialization for convolution layers
                    init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif isinstance(param, nn.Linear):
                    # Use Xavier initialization for fully connected layers
                    init.xavier_normal_(param)
                elif isinstance(param, nn.LSTM) or isinstance(param, nn.GRU):
                    # Use orthogonal initialization for RNN/LSTM weights
                    for weight_name, weight_param in param.named_parameters():
                        if 'weight' in weight_name:
                            init.orthogonal_(weight_param)
                        elif 'bias' in weight_name:
                            weight_param.data.fill_(0)
            elif 'bias' in name:
                # Set bias terms to 0
                param.data.fill_(0)
    
    def forward(self, joints, scene_xyz, gazes):
        """
        joints: (B, T, 23, 3)  History frames
        scene_xyz: (B, N, 3)   Scene point cloud
        gazes: (B, T, 1, 3)    Gaze trajectory
        return:
          predictions: (B, T+ΔT, 23, 3)
        """
        B, T, J, C = joints.shape
        _, N, _ = scene_xyz.shape
        assert T == self.input_seq_len, "The number of input frames must match config.input_seq_len"
        device = joints.device

        # ========== 1. Add Mask to Input Sequence and Future Frames ==========
        future_mask_joints = torch.zeros((B, self.output_seq_len, J, C), device=device)
        X_masked = torch.cat([joints, future_mask_joints], dim=1)  # (B, T+ΔT, 23, 3)

        # ========== 2. Encode Motion / Scene / Trajectory ==========
        F_X = self.pose_encoder(joints) 
        fp_features, bottleneck_feats = self.scene_encoder(scene_xyz.repeat(1,1,2))
        F_SS = fp_features.transpose(1, 2)
        F_traj_1 = self.traj_encoder(gazes)# (B, T, 64)
        F_traj_last = F_traj_1[:,[-1],:].repeat(1,10,1)
        F_traj = torch.cat([F_traj_1,F_traj_last],dim = 1)
        
        # ========== 3. Compute W: Based on gaze distance to scene point cloud ==========

        gaze_0 = gazes[:, 0, 0, :]  # (B, 3)
        dist = torch.norm(scene_xyz - gaze_0.unsqueeze(1), dim=-1, keepdim=True)
        dist_min, dist_max = dist.min(dim=1, keepdim=True)[0], dist.max(dim=1, keepdim=True)[0] 
        W = (dist - dist_min) / (dist_max - dist_min + 1e-8)  # (B, N, 1)
        W = 1-W
        
        
        
        # ========== 4. Multimodal Cross-Attention ==========
        # 4.1 Scene Point Cloud with W
        F_S = F_SS.clone()
        for layer in self.f_s_cross_att:
            F_S = layer(F_S, W)  # (B, N, 128)

        F_S_pooled = F_S.mean(dim=1, keepdim=True)
        F_S_tiled  = F_S_pooled.repeat(1, T+self.output_seq_len, 1)  # (B, T+ΔT, 128)

        # 4.2 Scene Point Cloud with Motion Sequence
        F_XS = F_X.clone()
        for layer in self.xs_cross_att:
            F_XS = layer(F_XS, F_S_tiled)  # (B, T+ΔT, 128)

        # 4.3 Scene Point Cloud with Trajectory Sequence
        F_TS = F_traj.clone()  
        for layer in self.ts_cross_att:
            F_TS = layer(F_TS, F_S_tiled)  # (B, T, 64)


        # ========== 5. Motion Prediction (Differentiable Priors) ==========
        # 5.1 Concatenate: [F_XS, F_X] -> Motion label logits
        cat_feat = torch.cat([F_XS, F_X], dim=-1)
        motion_logits = self.motion_type_mlp(cat_feat)
        motion_logits_last = motion_logits[:, -1, :]
        motion_label = torch.argmax(motion_logits_last, dim=-1)

        # 5.2 Convert motion_logits_last to embedding z
        z = self.label_to_z(motion_logits_last)  # [8, 128]

        # 5.3 Generate ending motion using differentiable prior network
        ending_motion_3d = self.ending_gen(z)    # (B, 23, 3)

        # Replace the last frame of X_masked with ending_motion_3d
        # This is just an example for batch=0, in actual use iterate over the batch or assign in X_masked
        X_masked[0, -1, :, :] = ending_motion_3d[0] # [8, 16, 23, 3]

        # ========== 6. Use TransformerFilling to Pad the Sequence ==========

        # Flatten -> proj -> Transformer -> inverse proj
        X_masked_flat = X_masked.reshape(B, self.seq_len, -1) # [8, 16, 69] 
        X_embed = self.pose_input_proj(X_masked_flat).transpose(0, 1) # [16, 8, 128]

        pred_embed = self.pose_transformer(X_embed, X_embed)  # (T, B, 128)
        pred_embed = pred_embed.transpose(0, 1)               # (B, T, 128)
        pred_flat  = self.pose_output_proj(pred_embed)        # (B, T, 69)
        pred_motion = pred_flat.reshape(B, self.seq_len, self.joint_num, 3)

        # ========== 7. Trajectory Prediction ==========

        H = scene_xyz[..., [2]]
        phi,_ =torch.max(H, dim=1, keepdim=True)
        phi /= 2
        M = (H > phi).float()
        F_SM = F_S * (1 - M)  # (B, N, 128)
        F_STS = F_TS.clone()  # [B,6,64]

        for layer in self.fm_cross_att:  # If matching N_d, it can be swapped with self.sts_cross_att
            F_STS = layer(F_STS, F_SM)
        
        F_STS = F_STS[:,6:,:]
        F_STS = torch.cat([F_traj_1,F_STS],dim = 1)

        # 7.x Use TransformerFilling to Predict Future Trajectory

        Traj_embed = self.traj_input_mlp(F_STS).transpose(0, 1)
        pred_traj_embed = self.traj_transformer(Traj_embed, Traj_embed) 
        pred_traj_embed = pred_traj_embed.transpose(0, 1) 
        pred_traj = self.traj_output_mlp(pred_traj_embed) # (B, T+ΔT, 3)
        pred_traj = pred_traj.reshape(B, self.seq_len, 1, 3)
        predictions = pred_motion.clone()
        predictions[:, :6, :, :] = joints
        distance = predictions[:,:,[0],:] - pred_traj
        predictions =  predictions - distance
        predictions[:, :, [0], :] = pred_traj  # (B, T+ΔT, 23, 3)

        for _ in range(1):
            predictions = self.motion_GCN(predictions)
            predictions[:, :6, :, :] = joints
            predictions = self.motion_GCN2(predictions)
        
        return predictions, motion_label
