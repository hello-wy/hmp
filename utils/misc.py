from pointnet2_ops import pointnet2_utils
import torch 



def downSample(points, npoints):
    """
    下采样点云，使用 FPS（最远点采样）。

    参数：
    - points: PyTorch 张量，形状为 [B, N, 3]，在 GPU 上
    - npoints: 目标下采样点数（整数）

    返回：
    - sampled_points: 下采样后的点云，形状为 [B, npoints, 3]，在 GPU 上
    """
    # 确保输入在 GPU 上，且形状为 [B, N, 3]
    assert points.dim() == 3 , "the shape of input point cloud should be [B, N, 3]"
    assert points.device.type == 'cuda', "point must be in GPU "

    # 使用 FPS 下采样，直接在 GPU 上操作
    batch_size, n_points, _ = points.shape
    new_idx = pointnet2_utils.furthest_point_sample(points, npoints)  # [B, npoints]
    
    # 使用高级索引提取下采样点
    sampled_points = points[torch.arange(batch_size)[:, None], new_idx]  # [B, npoints, 3]
    
    return sampled_points


def upSample(xyz,npoint):
    points = torch.from_numpy(xyz).float().cuda().unsqueeze(0)
    new_idx = pointnet2_utils.furthest_point_sample(points, npoint).squeeze().long().cpu()
    points = points.squeeze().cpu()[new_idx]
    
    return points