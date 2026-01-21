
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =============================================================================
# 1. 自注意力模块（SAGAN, ICML 2019）
# =============================================================================

class SelfAttention(nn.Module):
    """
    自注意力模块 - 让GAN关注缺陷区域

    论文：Self-Attention Generative Adversarial Networks (ICML 2019)
    作用：捕获长距离依赖，避免GAN只关注局部
    """

    def __init__(self, in_channels):
        super().__init__()

        # Query, Key, Value投影
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)

        # 可学习的缩放参数（从0开始，逐渐学习注意力）
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        # 计算Q, K, V
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B x (H*W) x C'
        proj_key = self.key_conv(x).view(B, -1, H * W)  # B x C' x (H*W)
        proj_value = self.value_conv(x).view(B, -1, H * W)  # B x C x (H*W)

        # 注意力矩阵
        attention = torch.bmm(proj_query, proj_key)  # B x (H*W) x (H*W)
        attention = F.softmax(attention, dim=-1)

        # 加权求和
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        # 残差连接
        out = self.gamma * out + x
        return out


# =============================================================================
# 2. U-Net生成器（Pix2Pix架构，CVPR 2017）
# =============================================================================

class UNetGenerator(nn.Module):
    """
    U-Net生成器 - 用于图像增强

    论文：Image-to-Image Translation with Conditional Adversarial Networks (CVPR 2017)

    架构特点：
    1. 编码器-解码器结构
    2. 跳跃连接（保留细节）
    3. 自注意力机制（关注缺陷）
    4. 残差输出（只生成增强差异）
    """

    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        """
        Args:
            in_channels: 输入通道数（RGB=3）
            out_channels: 输出通道数（RGB=3）
            ngf: 基础特征数（论文中用64）
        """
        super().__init__()

        # ========== 编码器（Encoder）==========
        # 逐步降采样，提取抽象特征

        self.enc1 = self._encoder_block(in_channels, ngf, normalize=False)  # 512 -> 512
        self.enc2 = self._encoder_block(ngf, ngf * 2)  # 512 -> 256
        self.enc3 = self._encoder_block(ngf * 2, ngf * 4)  # 256 -> 128
        self.enc4 = self._encoder_block(ngf * 4, ngf * 8)  # 128 -> 64

        # ========== 瓶颈层（Bottleneck）==========
        # 最深层特征 + 自注意力

        self.bottleneck = nn.Sequential(
            self._encoder_block(ngf * 8, ngf * 8),  # 64 -> 32
            SelfAttention(ngf * 8),  # 自注意力：让GAN关注缺陷
        )

        # ========== 解码器（Decoder）==========
        # 逐步上采样，恢复分辨率

        self.dec4 = self._decoder_block(ngf * 8, ngf * 8, dropout=0.5)  # 32 -> 64
        self.dec3 = self._decoder_block(ngf * 8 * 2, ngf * 4, dropout=0.5)  # 64 -> 128 (concat后通道翻倍)
        self.dec2 = self._decoder_block(ngf * 4 * 2, ngf * 2, dropout=0.5)  # 128 -> 256
        self.dec1 = self._decoder_block(ngf * 2 * 2, ngf)  # 256 -> 512

        # ========== 输出层 ==========
        # 生成残差图（不是完整图像！）

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, stride=2, padding=1),  # 512 -> 512
            nn.Tanh()  # 输出范围[-1, 1]
        )

    def _encoder_block(self, in_c, out_c, normalize=True):
        """编码器基础块：Conv -> BatchNorm -> LeakyReLU"""
        layers = [
            nn.Conv2d(in_c, out_c, 4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_c, out_c, dropout=0.0):
        """解码器基础块：TransposeConv -> BatchNorm -> Dropout -> ReLU"""
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]，值域[-1, 1]
        Returns:
            enhanced: 增强图像 [B, 3, H, W]
            residual: 残差图 [B, 3, H, W]（用于损失计算）
        """
        # 编码器（保存中间特征用于跳跃连接）
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 瓶颈层
        b = self.bottleneck(e4)

        # 解码器（带跳跃连接）
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)  # 拼接跳跃连接

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        # 生成残差
        residual = self.final(d1)

        # 最终输出 = 原图 + 0.3 * 残差（关键：防止偏离原图太远）
        enhanced = x + 0.3 * residual
        enhanced = torch.clamp(enhanced, -1, 1)

        return enhanced, residual


# =============================================================================
# 3. PatchGAN判别器（Pix2Pix架构）
# =============================================================================

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN判别器 - 判断图像的局部真实性

    论文：Image-to-Image Translation with Conditional Adversarial Networks (CVPR 2017)

    为什么用PatchGAN？
    - 传统判别器输出1个值（真/假）
    - PatchGAN输出NxN矩阵（每个patch真/假）
    - 优势：关注局部纹理，避免GAN模式崩溃
    """

    def __init__(self, in_channels=3, ndf=64):
        """
        Args:
            in_channels: 输入通道数（RGB=3）
            ndf: 基础特征数
        """
        super().__init__()

        # 使用光谱归一化（Spectral Normalization, ICLR 2018）稳定训练
        self.model = nn.Sequential(
            # Layer 1: 512 -> 256
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 256 -> 128
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1)
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 128 -> 64
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1)
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 64 -> 32
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1)
            ),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出层：32 -> 30 (PatchGAN输出)
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, 512, 512]
        Returns:
            output: 判别结果 [B, 1, 30, 30]（每个patch的真假概率）
        """
        return self.model(x)


# =============================================================================
# 4. 感知损失网络（VGG16）
# =============================================================================

class VGGPerceptualLoss(nn.Module):
    """
    感知损失 - 让增强图像在语义上接近原图

    论文：Perceptual Losses for Real-Time Style Transfer (ECCV 2016)

    为什么需要感知损失？
    - L1/L2损失只关注像素差异
    - 感知损失关注语义特征
    - 防止GAN生成"看起来假"的图像
    """

    def __init__(self):
        super().__init__()

        # 使用预训练VGG16的conv3_3层
        try:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval()

            # 冻结参数
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        except:
            print("警告：无法加载VGG16，感知损失将被禁用")
            self.feature_extractor = None

    def forward(self, x, y):
        """
        计算感知损失

        Args:
            x, y: 两张图像 [B, 3, H, W]
        Returns:
            loss: 感知损失标量
        """
        if self.feature_extractor is None:
            return torch.tensor(0.0, device=x.device)

        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        return F.mse_loss(x_features, y_features)


# =============================================================================
# 5. DA-CGAN完整训练框架（🔧 修复enhance函数）
# =============================================================================

class DefectAwareConditionalGAN:
    """
    缺陷感知条件GAN - 完整训练和推理框架

    替换原有的HierarchicalDifferentialEvolution类

    损失函数组合：
    1. 对抗损失（Adversarial Loss）：让生成图像看起来真实
    2. L1损失（L1 Loss）：保持图像结构
    3. 感知损失（Perceptual Loss）：保持语义内容
    4. 残差正则化：防止过度增强
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"DA-CGAN 使用设备: {self.device}")

        # 初始化网络
        self.generator = UNetGenerator(in_channels=3, out_channels=3, ngf=64).to(device)
        self.discriminator = PatchGANDiscriminator(in_channels=3, ndf=64).to(device)
        self.perceptual_loss = VGGPerceptualLoss().to(device)

        # 优化器（使用Adam，论文标准配置）
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999)  # Pix2Pix论文的超参数
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999)
        )

        # 损失权重（可调）
        self.lambda_l1 = 30.0  # L1损失权重
        self.lambda_perceptual = 5.0  # 感知损失权重
        self.lambda_residual = 2.0  # 残差正则化权重

        # 训练状态
        self.is_trained = False

    def enhance(self, image):
        """
        🔧 修复版：推理函数，使用训练好的GAN增强图像

        修复内容：正确的归一化和反归一化流程

        Args:
            image: NumPy图像，BGR格式，uint8类型，shape=(H,W,3)，值域[0,255]
        Returns:
            enhanced: 增强后的图像，BGR格式，uint8类型，shape=(H,W,3)
        """
        import cv2
        import numpy as np
        import torch

        # 保存原始尺寸
        original_h, original_w = image.shape[:2]

        # 1. 调整到512x512（GAN训练尺寸）
        if original_h != 512 or original_w != 512:
            image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = image.copy()

        # 2. BGR → RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # 3. 🔥 归一化到 [-1, 1]
        image_float = image_rgb.astype(np.float32)
        image_normalized = (image_float / 127.5) - 1.0  # [0,255] → [-1,1]

        # 4. numpy → tensor: (H,W,C) → (1,C,H,W)
        input_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # 5. 前向传播
        self.generator.eval()
        with torch.no_grad():
            enhanced_tensor, _ = self.generator(input_tensor)

        # 6. tensor → numpy: (1,C,H,W) → (H,W,C)
        enhanced_tensor = enhanced_tensor.squeeze(0).permute(1, 2, 0)
        enhanced_np = enhanced_tensor.cpu().numpy()

        # 7. 🔥 反归一化：[-1, 1] → [0, 255]
        enhanced_np = (enhanced_np + 1.0) * 127.5  # [-1,1] → [0,255]
        enhanced_np = np.clip(enhanced_np, 0, 255).astype(np.uint8)

        # 8. RGB → BGR
        enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)

        # 9. 恢复原始尺寸
        if original_h != 512 or original_w != 512:
            enhanced_bgr = cv2.resize(enhanced_bgr, (original_w, original_h),
                                     interpolation=cv2.INTER_LINEAR)

        return enhanced_bgr

    def _to_tensor(self, image):
        """
        NumPy图像 → PyTorch Tensor（训练时使用）

        Args:
            image: [H, W, 3], uint8, [0, 255], BGR
        Returns:
            tensor: [1, 3, H, W], float32, [-1, 1]
        """
        # BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize to [-1, 1]
        tensor = torch.from_numpy(image_rgb).float() / 127.5 - 1.0
        # HWC → CHW
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _to_numpy(self, tensor):
        """
        PyTorch Tensor → NumPy图像（训练时使用）

        Args:
            tensor: [1, 3, H, W], float32, [-1, 1]
        Returns:
            image: [H, W, 3], uint8, [0, 255], BGR
        """
        # [-1, 1] → [0, 255]
        image = ((tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
        # RGB → BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image_bgr

    def train_step(self, real_image):
        """
        单步训练（一次迭代）

        Args:
            real_image: NumPy图像 [H, W, 3]
        Returns:
            losses: 损失字典
        """
        real_tensor = self._to_tensor(real_image)

        # ========== 训练判别器 ==========
        self.d_optimizer.zero_grad()

        # 生成假图像
        with torch.no_grad():
            fake_tensor, _ = self.generator(real_tensor)

        # 判别器判断真图
        pred_real = self.discriminator(real_tensor)
        loss_d_real = F.mse_loss(pred_real, torch.ones_like(pred_real))

        # 判别器判断假图
        pred_fake = self.discriminator(fake_tensor)
        loss_d_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

        # 判别器总损失
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        self.d_optimizer.step()

        # ========== 训练生成器 ==========
        self.g_optimizer.zero_grad()

        # 生成假图像
        fake_tensor, residual = self.generator(real_tensor)

        # 1. 对抗损失（欺骗判别器）
        pred_fake = self.discriminator(fake_tensor)
        loss_g_adv = F.mse_loss(pred_fake, torch.ones_like(pred_fake))

        # 2. L1损失（保持结构）
        loss_g_l1 = F.l1_loss(fake_tensor, real_tensor) * self.lambda_l1

        # 3. 感知损失（保持语义）
        loss_g_perceptual = self.perceptual_loss(fake_tensor, real_tensor) * self.lambda_perceptual

        # 4. 残差正则化（防止过度增强）
        loss_g_residual = torch.mean(torch.abs(residual)) * self.lambda_residual

        # 生成器总损失
        loss_g = loss_g_adv + loss_g_l1 + loss_g_perceptual + loss_g_residual
        loss_g.backward()
        self.g_optimizer.step()

        return {
            'd_loss': loss_d.item(),
            'g_loss': loss_g.item(),
            'g_adv': loss_g_adv.item(),
            'g_l1': loss_g_l1.item() / self.lambda_l1,
            'g_perceptual': loss_g_perceptual.item() / self.lambda_perceptual,
            'g_residual': loss_g_residual.item() / self.lambda_residual,
        }

    def train(self, train_images, num_epochs=50, save_interval=10):
        """
        训练GAN

        Args:
            train_images: 训练图像列表 [N, H, W, 3]
            num_epochs: 训练轮数
            save_interval: 保存间隔
        """
        print(f"开始训练 DA-CGAN，共 {num_epochs} 轮，{len(train_images)} 张图像")

        for epoch in range(num_epochs):
            epoch_losses = []

            for idx, image in enumerate(train_images):
                losses = self.train_step(image)
                epoch_losses.append(losses)

                if (idx + 1) % 10 == 0:
                    avg_g_loss = np.mean([l['g_loss'] for l in epoch_losses[-10:]])
                    avg_d_loss = np.mean([l['d_loss'] for l in epoch_losses[-10:]])
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{idx+1}/{len(train_images)}], "
                          f"G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")

            # Epoch总结
            print(f"\nEpoch {epoch+1} 完成:")
            print(f"  生成器损失: {np.mean([l['g_loss'] for l in epoch_losses]):.4f}")
            print(f"  判别器损失: {np.mean([l['d_loss'] for l in epoch_losses]):.4f}")
            print(f"  对抗损失: {np.mean([l['g_adv'] for l in epoch_losses]):.4f}")
            print(f"  L1损失: {np.mean([l['g_l1'] for l in epoch_losses]):.4f}")
            print(f"  感知损失: {np.mean([l['g_perceptual'] for l in epoch_losses]):.4f}")

            # 定期保存模型
            if (epoch + 1) % save_interval == 0:
                self.save_model(f'dacgan_epoch_{epoch+1}.pth')

        self.is_trained = True
        print("\n✓ DA-CGAN训练完成！")

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
        }, path)
        print(f"模型已保存: {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.is_trained = True
        print(f"模型已加载: {path}")


# =============================================================================
# 6. 单独的训练脚本（首次使用时需要运行）
# =============================================================================

def train_dacgan_standalone(base_dir=r"D:\datasets\dataset\mydata", num_epochs=10):
    """
    单独训练DA-CGAN的脚本

    Args:
        base_dir: 数据集根目录
        num_epochs: 训练轮数
    """
    print("=" * 80)
    print("DA-CGAN 训练脚本")
    print("=" * 80)

    # 1. 加载训练数据
    train_dir = os.path.join(base_dir, "images", "train")

    print(f"\n加载训练数据: {train_dir}")
    train_images = []
    for filename in os.listdir(train_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(train_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # 调整到512x512（GAN训练需要固定尺寸）
                img = cv2.resize(img, (512, 512))
                train_images.append(img)

    print(f"✓ 加载了 {len(train_images)} 张训练图像")

    if len(train_images) == 0:
        print("错误：未找到训练图像！")
        return None

    # 2. 初始化DA-CGAN
    dacgan = DefectAwareConditionalGAN()

    # 3. 训练GAN
    print("\n开始训练...")
    dacgan.train(
        train_images=train_images,
        num_epochs=num_epochs,
        save_interval=5
    )

    # 4. 保存最终模型
    dacgan.save_model('dacgan_pretrained.pth')
    print("\n✓ 训练完成！模型已保存为 dacgan_pretrained.pth")
    print("   现在可以在主程序中使用此模型进行增强")

    return dacgan


# =============================================================================
# 7. 快速测试代码（验证GAN是否正常工作）
# =============================================================================

def quick_test_dacgan():
    """
    快速测试DA-CGAN是否正常工作

    使用少量图像快速验证
    """
    print("\n" + "=" * 80)
    print("DA-CGAN 快速测试")
    print("=" * 80)

    # 1. 创建测试数据（随机噪声图像）
    print("\n1. 创建测试数据...")
    test_images = []
    for i in range(5):  # 只用5张图测试
        # 生成512x512的随机图像
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_images.append(img)
    print(f"✓ 创建了 {len(test_images)} 张测试图像")

    # 2. 初始化DA-CGAN
    print("\n2. 初始化DA-CGAN...")
    dacgan = DefectAwareConditionalGAN()
    print("✓ DA-CGAN初始化完成")

    # 3. 快速训练（只训练1轮）
    print("\n3. 快速训练（1轮）...")
    dacgan.train(test_images, num_epochs=1, save_interval=1)

    # 4. 测试推理
    print("\n4. 测试推理...")
    enhanced = dacgan.enhance(test_images[0])
    print(f"✓ 增强成功！输入形状: {test_images[0].shape}, 输出形状: {enhanced.shape}")

    # 验证输出范围
    print(f"  输入范围: [{test_images[0].min()}, {test_images[0].max()}]")
    print(f"  输出范围: [{enhanced.min()}, {enhanced.max()}]")
    print(f"  输出均值: {enhanced.mean():.1f}")

    # 5. 保存测试结果
    cv2.imwrite('test_original.jpg', test_images[0])
    cv2.imwrite('test_enhanced.jpg', enhanced)
    print("✓ 测试图像已保存：test_original.jpg, test_enhanced.jpg")

    print("\n" + "=" * 80)
    print("✓ DA-CGAN测试通过！")
    print("  所有组件正常工作，可以开始正式训练")
    print("=" * 80)

    return dacgan


# =============================================================================
# 8. 主程序使用示例
# =============================================================================

if __name__ == "__main__":
    """
    使用说明：
    
    模式1：训练DA-CGAN（首次使用）
    --------------------------------
    python dacgan.py train
    
    模式2：快速测试
    --------------------------------
    python dacgan.py test
    
    模式3：在其他脚本中使用
    --------------------------------
    from dacgan import DefectAwareConditionalGAN
    
    dacgan = DefectAwareConditionalGAN()
    dacgan.load_model('dacgan_pretrained.pth')
    enhanced = dacgan.enhance(image)
    """

    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == 'train':
            # 训练模式
            print("\n模式：训练DA-CGAN")
            dacgan = train_dacgan_standalone(num_epochs=10)

        elif mode == 'test':
            # 测试模式
            print("\n模式：快速测试DA-CGAN")
            dacgan = quick_test_dacgan()

        else:
            print(f"未知模式: {mode}")
            print("使用方法: python dacgan.py [train|test]")

    else:
        # 默认：显示帮助信息
        print("\n" + "=" * 80)
        print("DA-CGAN - Defect-Aware Conditional GAN")
        print("=" * 80)
        print("\n使用方法：")
        print("  python dacgan.py train    # 训练GAN")
        print("  python dacgan.py test     # 快速测试")
        print("\n在其他脚本中使用：")
        print("  from dacgan import DefectAwareConditionalGAN")
        print("  dacgan = DefectAwareConditionalGAN()")
        print("  dacgan.load_model('dacgan_pretrained.pth')")
        print("  enhanced = dacgan.enhance(image)")
        print("=" * 80)

print("\n" + "=" * 80)
print("DA-CGAN 模块加载完成！")
print("=" * 80)
print("\n快速开始：")
print("1. 训练：python dacgan.py train")
print("2. 测试：python dacgan.py test")
print("3. 集成：from dacgan import DefectAwareConditionalGAN")
