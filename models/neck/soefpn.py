"""
SOEFPN - 小目标增强特征金字塔网络
Small Object Enhanced Feature Pyramid Network
"""
import torch
import torch.nn as nn


class CSPMFOK(nn.Module):
    """
    跨阶段部分连接多尺度频率感知全向卷积模块
    Cross-Stage Partial Multi-scale Frequency-aware Omni-Kernel
    """
    def __init__(self, in_channels, out_channels):
        super(CSPMFOK, self).__init__()
        
        # TODO: 实现CSPMFOK模块
        # 1. 全向多尺度卷积
        # 2. 双域通道注意力(DDCA)
        # 3. 频域门控调制(FDGM)
        
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        print(f'CSPMFOK初始化: in={in_channels}, out={out_channels}')
    
    def forward(self, x):
        # TODO: 实现前向传播
        return self.conv(x)


class HMSAF(nn.Module):
    """
    层次化多尺度注意力融合模块
    Hierarchical Multi-Scale Attention Fusion
    """
    def __init__(self, channels):
        super(HMSAF, self).__init__()
        
        # TODO: 实现HMSAF模块
        # 1. 局部-全局双路径注意力
        # 2. 语义原型引导
        # 3. 特征调制
        
        self.conv = nn.Conv2d(channels * 2, channels, 1)
        print(f'HMSAF初始化: channels={channels}')
    
    def forward(self, x1, x2):
        # TODO: 实现前向传播
        x = torch.cat([x1, x2])
        return self.conv(x)


class SPDConv(nn.Module):
    """
    空间到深度卷积
    Space-to-Depth Convolution
    """
    def __init__(self, in_channels, out_channels):
        super(SPDConv, self).__init__()
        
        # Space-to-Depth操作
        self.conv = nn.Conv2d(in_channels * 4, out_channels, 1)
    
    def forward(self, x):
        # 将空间维度转换为通道维度
        B, C, H, W = x.shape
        x = x.view(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C*4, H//2, W//2)
        x = self.conv(x)
        return x


class SOEFPN(nn.Module):
    """
    小目标增强特征金字塔网络
    
    Args:
        in_channels (list): 输入通道数 [C3, C4, C5]
        out_channels (int): 输出通道数
        use_cspmfok (bool): 是否使用CSPMFOK模块
        use_hmsaf (bool): 是否使用HMSAF模块
        use_spd_conv (bool): 是否使用SPDConv
    """
    def __init__(self, in_channels=[128, 256, 512], out_channels=256,
                 use_cspmfok=True, use_hmsaf=True, use_spd_conv=True):
        super(SOEFPN, self).__init__()
        
        self.use_cspmfok = use_cspmfok
        self.use_hmsaf = use_hmsaf
        self.use_spd_conv = use_spd_conv
        
        # 通道对齐
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels
        ])
        
        # SPDConv用于P2->P3
        if use_spd_conv:
            self.spd_conv = SPDConv(in_channels[0]//2, out_channels)
        
        # CSPMFOK模块
        if use_cspmfok:
            self.cspmfok_modules = nn.ModuleList([
                CSPMFOK(out_channels, out_channels)
                for _ in range(len(in_channels))
            ])
        
        # HMSAF模块
        if use_hmsaf:
            self.hmsaf_modules = nn.ModuleList([
                HMSAF(out_channels)
                for _ in range(len(in_channels) - 1)
            ])
        
        # 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        print(f'SOEFPN初始化完成: in_channels={in_channels}, out_channels={out_channels}')
    
    def forward(self, features):
        """
        前向传播
        
        Args:
            features (list): 主干网络输出 [P3, P4, P5]
        
        Returns:
            outputs (list): 融合后的特征 [P3, P4, P5]
        """
        # 通道对齐
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # 自顶向下融合
        # P5
        p5 = laterals[2]
        if self.use_cspmfok:
            p5 = self.cspmfok_modules[2](p5)
        
        # P4
        p4 = laterals[1]
        if self.use_hmsaf:
            p4 = self.hmsaf_modules[1](p4, self.upsample(p5))
        else:
            p4 = p4 + self.upsample(p5)
        if self.use_cspmfok:
            p4 = self.cspmfok_modules[1](p4)
        
        # P3
        p3 = laterals[0]
        if self.use_hmsaf:
            p3 = self.hmsaf_modules[0](p3, self.upsample(p4))
        else:
            p3 = p3 + self.upsample(p4)
        if self.use_cspmfok:
            p3 = self.cspmfok_modules[0](p3)
        
        # TODO: 如果使用SPDConv，将P2特征注入P3
        # if self.use_spd_conv:
        #     p2_features = self.spd_conv(p2)
        #     p3 = p3 + p2_features
        
        return [p3, p4, p5]


if __name__ == '__main__':
    # 测试SOEFPN
    model = SOEFPN()
    
    # 模拟主干网络输出
    p3 = torch.randn(2, 128, 80, 80)
    p4 = torch.randn(2, 256, 40, 40)
    p5 = torch.randn(2, 512, 20, 20)
    
    features = [p3, p4, p5]
    outputs = model(features)
    
    print(f"输入特征:")
    for i, feat in enumerate(features):
        print(f"  P{i+3}: {feat.shape}")
    
    print(f"输出特征:")
    for i, feat in enumerate(outputs):
        print(f"  P{i+3}: {feat.shape}")