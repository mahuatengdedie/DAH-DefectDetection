
import cv2
import numpy as np
import os
from scipy.optimize import differential_evolution
from scipy import stats
import warnings
import argparse

warnings.filterwarnings('ignore')

# 导入DA-CGAN（确保dacgan.py在同一目录）
try:
    from dacgan import DefectAwareConditionalGAN

    DACGAN_AVAILABLE = True
except ImportError:
    print("⚠️  警告：未找到dacgan.py，DA-CGAN功能将不可用")
    print("    请确保dacgan.py在同一目录下")
    DACGAN_AVAILABLE = False


# =============================================================================
# 第一部分：学术级特征提取器 AGFPN（保留）
# =============================================================================

class AdaptiveKernelGenerator:
    """自适应卷积核生成器"""

    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
        self.base_kernels = self._initialize_base_kernels()

    def _initialize_base_kernels(self):
        """初始化基础核库（参数化表示）"""
        kernels = {}

        def gabor_kernel(theta, frequency, sigma):
            """参数化Gabor核"""
            size = self.kernel_size
            kernel = np.zeros((size, size))
            center = size // 2

            for i in range(size):
                for j in range(size):
                    x = (i - center) * np.cos(theta) + (j - center) * np.sin(theta)
                    y = -(i - center) * np.sin(theta) + (j - center) * np.cos(theta)
                    kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) * \
                                   np.cos(2 * np.pi * frequency * x)
            return kernel / (np.sum(np.abs(kernel)) + 1e-10)

        kernels['gabor_generator'] = gabor_kernel

        def directional_derivative(theta, order):
            """参数化方向导数核"""
            size = self.kernel_size
            kernel = np.zeros((size, size))
            center = size // 2

            for i in range(size):
                for j in range(size):
                    x = i - center
                    y = j - center
                    xr = x * np.cos(theta) + y * np.sin(theta)
                    if order == 1:
                        kernel[i, j] = xr
                    elif order == 2:
                        kernel[i, j] = xr ** 2 - 1
            return kernel / (np.sum(np.abs(kernel)) + 1e-10)

        kernels['derivative_generator'] = directional_derivative

        def adaptive_log(sigma):
            """参数化LoG核"""
            size = self.kernel_size
            kernel = np.zeros((size, size))
            center = size // 2

            for i in range(size):
                for j in range(size):
                    x = i - center
                    y = j - center
                    r2 = x ** 2 + y ** 2
                    kernel[i, j] = -(1 / (np.pi * sigma ** 4)) * \
                                   (1 - r2 / (2 * sigma ** 2)) * \
                                   np.exp(-r2 / (2 * sigma ** 2))
            return kernel / (np.sum(np.abs(kernel)) + 1e-10)

        kernels['log_generator'] = adaptive_log
        return kernels

    def analyze_local_statistics(self, patch):
        """分析局部图像块的统计特性"""
        stats_dict = {}

        grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        orientation = np.arctan2(grad_y, grad_x)

        hist, bins = np.histogram(orientation, bins=8, range=(-np.pi, np.pi), weights=magnitude)
        dominant_direction = bins[np.argmax(hist)]

        stats_dict['dominant_direction'] = dominant_direction
        stats_dict['edge_strength'] = np.mean(magnitude)

        autocorr = cv2.filter2D(patch.astype(np.float32), -1, patch.astype(np.float32))
        center = autocorr.shape[0] // 2
        autocorr[center - 1:center + 2, center - 1:center + 2] = 0
        peak_distance = np.unravel_index(np.argmax(autocorr), autocorr.shape)
        dominant_frequency = 1.0 / (np.sqrt((peak_distance[0] - center) ** 2 +
                                            (peak_distance[1] - center) ** 2) + 1)

        stats_dict['dominant_frequency'] = dominant_frequency
        stats_dict['local_variance'] = np.var(patch)

        return stats_dict

    def generate_adaptive_kernel(self, image_patch):
        """根据图像块生成自适应卷积核"""
        stats_dict = self.analyze_local_statistics(image_patch)

        edge_strength = stats_dict['edge_strength']
        variance = stats_dict['local_variance']

        if edge_strength > 10:
            theta = stats_dict['dominant_direction']
            frequency = stats_dict['dominant_frequency']
            sigma = 1.0
            kernel = self.base_kernels['gabor_generator'](theta, frequency, sigma)
        elif variance < 50:
            sigma = 1.5
            kernel = self.base_kernels['log_generator'](sigma)
        else:
            theta = stats_dict['dominant_direction']
            kernel = self.base_kernels['derivative_generator'](theta, order=2)

        return kernel


class AttentionGuidedFPN:
    """注意力引导的特征金字塔网络"""

    def __init__(self, min_scale=0.5, max_scale=2.0, num_scales=5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_scales = num_scales
        self.kernel_generator = AdaptiveKernelGenerator()

    def select_adaptive_scales(self, image):
        """自适应选择最优尺度集合（基于FFT频率分析）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        f_transform = np.fft.fft2(gray)
        f_shifted = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shifted)

        h, w = gray.shape
        cy, cx = h // 2, w // 2

        frequency_bands = []
        for r in range(5, min(h, w) // 2, 10):
            mask = np.zeros((h, w))
            y, x = np.ogrid[:h, :w]
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask[(distance >= r - 5) & (distance < r + 5)] = 1
            band_energy = np.sum(magnitude * mask)
            frequency_bands.append((r, band_energy))

        frequency_bands.sort(key=lambda x: x[1], reverse=True)

        max_radius = min(h, w) // 2
        selected_scales = []
        for r, energy in frequency_bands[:self.num_scales]:
            scale = self.min_scale + (self.max_scale - self.min_scale) * (1 - r / max_radius)
            selected_scales.append(scale)

        if 1.0 not in selected_scales:
            selected_scales[0] = 1.0

        return sorted(selected_scales)

    def compute_scale_attention(self, feature_maps):
        """计算每个尺度的注意力权重"""
        n_scales = len(feature_maps)
        attentions = np.zeros(n_scales)

        for i, feature in enumerate(feature_maps):
            hist, _ = np.histogram(feature.flatten(), bins=50)
            hist = hist + 1e-10
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob))

            grad_x = cv2.Sobel(feature, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(feature, cv2.CV_64F, 0, 1, ksize=3)
            gradient_strength = np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2))

            attentions[i] = entropy * 0.6 + gradient_strength * 0.4

        attentions = np.exp(attentions - np.max(attentions))
        attentions = attentions / np.sum(attentions)

        return attentions

    def bidirectional_fusion(self, feature_maps, scales):
        """双向特征融合"""
        n_scales = len(feature_maps)

        bottom_up_features = [feature_maps[0]]
        for i in range(1, n_scales):
            current = feature_maps[i]
            previous = bottom_up_features[-1]

            if previous.shape != current.shape:
                previous_resized = cv2.resize(previous, (current.shape[1], current.shape[0]),
                                              interpolation=cv2.INTER_AREA)
            else:
                previous_resized = previous

            fused = 0.6 * current + 0.4 * previous_resized
            bottom_up_features.append(fused)

        top_down_features = [bottom_up_features[-1]]
        for i in range(n_scales - 2, -1, -1):
            current = bottom_up_features[i]
            previous = top_down_features[-1]

            if previous.shape != current.shape:
                previous_resized = cv2.resize(previous, (current.shape[1], current.shape[0]),
                                              interpolation=cv2.INTER_LINEAR)
            else:
                previous_resized = previous

            fused = 0.6 * current + 0.4 * previous_resized
            top_down_features.insert(0, fused)

        return top_down_features

    def extract_features(self, image):
        """完整的特征提取流程"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        scales = self.select_adaptive_scales(gray)

        feature_maps = []
        original_size = gray.shape

        for scale in scales:
            if scale != 1.0:
                new_size = (int(original_size[1] * scale), int(original_size[0] * scale))
                new_size = (max(3, new_size[0]), max(3, new_size[1]))
                scaled_img = cv2.resize(gray, new_size, interpolation=cv2.INTER_LINEAR)
            else:
                scaled_img = gray

            kernel = self.kernel_generator.generate_adaptive_kernel(scaled_img)
            feature = cv2.filter2D(scaled_img.astype(np.float32), -1, kernel)

            if scale != 1.0:
                feature = cv2.resize(feature, (original_size[1], original_size[0]),
                                     interpolation=cv2.INTER_LINEAR)

            feature_maps.append(feature)

        fused_features = self.bidirectional_fusion(feature_maps, scales)
        attention_weights = self.compute_scale_attention(fused_features)

        final_feature = np.zeros_like(fused_features[0])
        for feature, weight in zip(fused_features, attention_weights):
            final_feature += feature * weight

        return final_feature, attention_weights, scales


class CooperativeFeatureEnhancement:
    """协同特征增强（图卷积）"""

    def __init__(self, block_size=32, overlap=16):
        self.block_size = block_size
        self.overlap = overlap

    def construct_feature_graph(self, image):
        """构建特征图"""
        h, w = image.shape[:2]

        actual_block_size = min(self.block_size, h // 2, w // 2)
        if actual_block_size < 8:
            actual_block_size = min(h, w)

        actual_overlap = min(self.overlap, actual_block_size // 2)
        stride = max(1, actual_block_size - actual_overlap)

        blocks = []
        positions = []

        for i in range(0, h - actual_block_size + 1, stride):
            for j in range(0, w - actual_block_size + 1, stride):
                block = image[i:i + actual_block_size, j:j + actual_block_size]
                if len(block.shape) == 3:
                    block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                blocks.append(block)
                positions.append((i, j))

        if len(blocks) == 0:
            if len(image.shape) == 3:
                block = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                block = image.copy()
            blocks.append(block)
            positions.append((0, 0))

        n_blocks = len(blocks)
        adjacency = np.zeros((n_blocks, n_blocks))

        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                dist = np.sqrt((positions[i][0] - positions[j][0]) ** 2 +
                               (positions[i][1] - positions[j][1]) ** 2)
                if dist < 2 * stride:
                    weight = np.exp(-dist / (stride + 1e-10))
                    adjacency[i, j] = weight
                    adjacency[j, i] = weight

        return blocks, adjacency, positions

    def graph_convolution_propagation(self, blocks, adjacency, num_iterations=3):
        """图卷积传播"""
        n_blocks = len(blocks)

        if n_blocks == 1:
            return np.array([1.0])

        features = np.zeros((n_blocks, 4))

        for i, block in enumerate(blocks):
            features[i, 0] = np.mean(block)
            features[i, 1] = np.std(block)
            features[i, 2] = stats.skew(block.flatten())
            features[i, 3] = stats.kurtosis(block.flatten())

        feature_std = np.std(features, axis=0)
        feature_std[feature_std < 1e-6] = 1.0
        features = (features - np.mean(features, axis=0)) / feature_std

        degree = np.sum(adjacency, axis=1)
        degree[degree < 1e-10] = 1.0
        D_sqrt_inv = np.diag(1.0 / np.sqrt(degree))

        A_norm = D_sqrt_inv @ adjacency @ D_sqrt_inv

        current_features = features.copy()
        for iteration in range(num_iterations):
            propagated = A_norm @ current_features
            propagated = np.maximum(propagated, 0)
            current_features = 0.7 * current_features + 0.3 * propagated

        enhancement_weights = np.linalg.norm(current_features, axis=1)
        max_weight = np.max(enhancement_weights)
        if max_weight < 1e-10:
            enhancement_weights = np.ones(n_blocks)
        else:
            enhancement_weights = enhancement_weights / max_weight

        return enhancement_weights

    def apply_cooperative_enhancement(self, image):
        """应用协同增强"""
        blocks, adjacency, positions = self.construct_feature_graph(image)
        weights = self.graph_convolution_propagation(blocks, adjacency)

        if len(image.shape) == 3:
            enhanced = image.astype(np.float32).copy()
        else:
            enhanced = image.astype(np.float32)

        h, w = image.shape[:2]
        actual_block_size = min(self.block_size, h // 2, w // 2)
        if actual_block_size < 8:
            actual_block_size = min(h, w)

        actual_overlap = min(self.overlap, actual_block_size // 2)
        stride = max(1, actual_block_size - actual_overlap)

        weight_map = np.ones(image.shape[:2], dtype=np.float32)
        count_map = np.zeros(image.shape[:2], dtype=np.float32)

        for idx, (i, j) in enumerate(positions):
            end_i = min(i + actual_block_size, h)
            end_j = min(j + actual_block_size, w)
            weight_map[i:end_i, j:end_j] += weights[idx]
            count_map[i:end_i, j:end_j] += 1

        count_map[count_map < 1] = 1.0
        weight_map = weight_map / count_map
        weight_map = cv2.normalize(weight_map, None, 0.8, 1.2, cv2.NORM_MINMAX)

        if len(image.shape) == 3:
            for c in range(3):
                enhanced[:, :, c] *= weight_map
        else:
            enhanced *= weight_map

        return np.clip(enhanced, 0, 255).astype(np.uint8)


class AcademicFeatureExtractor:
    """学术级特征提取器 - AGFPN"""

    def __init__(self):
        self.fpn = AttentionGuidedFPN()
        self.cooperative = CooperativeFeatureEnhancement()

    def extract_and_enhance(self, image, enhancement_factor=0.2):
        """完整的特征提取和增强流程"""
        fpn_feature, attention_weights, scales = self.fpn.extract_features(image)

        fpn_feature_uint8 = np.clip(fpn_feature, 0, 255).astype(np.uint8)
        cooperative_enhanced = self.cooperative.apply_cooperative_enhancement(fpn_feature_uint8)

        if len(image.shape) == 3:
            enhanced_channels = []
            for c in range(3):
                channel = image[:, :, c].astype(np.float32)
                enhanced_channel = channel + enhancement_factor * cooperative_enhanced.astype(np.float32)
                enhanced_channel = np.clip(enhanced_channel, 0, 255)
                enhanced_channels.append(enhanced_channel)
            result = np.stack(enhanced_channels, axis=-1).astype(np.uint8)
        else:
            enhanced = image.astype(np.float32) + enhancement_factor * cooperative_enhanced.astype(np.float32)
            result = np.clip(enhanced, 0, 255).astype(np.uint8)

        return result


# =============================================================================
# 第二部分：分层差分进化算法 HDE（保留）
# =============================================================================

class HierarchicalDifferentialEvolution:
    """分层差分进化算法"""

    @staticmethod
    def hierarchical_optimize(image, max_generations=30):
        """分层优化策略"""
        global_bounds = [(0.5, 2.0), (-50, 50), (0.5, 2.0), (0.1, 2.0), (0.0, 1.0)]
        global_result = differential_evolution(
            lambda params: HierarchicalDifferentialEvolution.coarse_fitness(params, image),
            global_bounds, maxiter=10, popsize=15, mutation=(0.7, 1.2)
        )

        best_global = global_result.x
        fine_bounds = [
            (max(0.5, best_global[0] - 0.3), min(2.0, best_global[0] + 0.3)),
            (max(-50, best_global[1] - 20), min(50, best_global[1] + 20)),
            (max(0.5, best_global[2] - 0.3), min(2.0, best_global[2] + 0.3)),
            (max(0.1, best_global[3] - 0.5), min(2.0, best_global[3] + 0.5)),
            (max(0.0, best_global[4] - 0.3), min(1.0, best_global[4] + 0.3))
        ]

        fine_result = differential_evolution(
            lambda params: HierarchicalDifferentialEvolution.fine_fitness(params, image),
            fine_bounds, maxiter=20, popsize=8, mutation=(0.3, 0.7)
        )

        return fine_result.x

    @staticmethod
    def coarse_fitness(params, image):
        """粗调适应度函数"""
        small_image = cv2.resize(image, (128, 128))
        return HierarchicalDifferentialEvolution.multi_objective_fitness(params, small_image)

    @staticmethod
    def fine_fitness(params, image):
        """细调适应度函数"""
        return HierarchicalDifferentialEvolution.multi_objective_fitness(params, image)

    @staticmethod
    def multi_objective_fitness(params, image):
        """多目标适应度函数"""
        alpha, beta, gamma, sigma, sharpen_strength = params

        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        enhanced = HierarchicalDifferentialEvolution.apply_gamma(enhanced, gamma)

        if sigma > 0.1:
            enhanced = cv2.GaussianBlur(enhanced, (5, 5), sigma)

        if sharpen_strength > 0:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) * sharpen_strength
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        if len(enhanced.shape) == 3:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            gray = enhanced

        variance = np.var(gray)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.mean(np.abs(sobel_x) + np.abs(sobel_y))

        kernel = np.ones((3, 3)) / 9
        smoothed = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        texture = np.mean((gray.astype(np.float32) - smoothed) ** 2)

        score = variance * 0.4 + gradient * 0.4 + texture * 0.2
        return -score

    @staticmethod
    def apply_gamma(image, gamma):
        """应用伽马校正"""
        if gamma <= 0:
            gamma = 1.0
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)


# =============================================================================
# 第三部分：主处理流程
# =============================================================================

def resize_image(image, width=None, height=None):
    if width is None and height is None:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def normalize_image(image):
    return image.astype(np.float32) / 255.0


def process_dataset(base_dir, subset, config, suffix="_FINAL"):
    """
    数据集处理函数

    处理流程：DA-CGAN → AGFPN → HDE
    """
    input_dir = os.path.join(base_dir, "images", subset)
    output_dir = os.path.join(base_dir, "images", f"{subset}{suffix}")
    os.makedirs(output_dir, exist_ok=True)

    # ========== 初始化模块 ==========

    # 1. AGFPN特征提取器
    academic_extractor = None
    if config.get("use_feature_extraction", False):
        print("  初始化AGFPN特征提取器...")
        academic_extractor = AcademicFeatureExtractor()

    # 2. DA-CGAN（替换MSA）
    dacgan = None
    if config.get("use_dacgan", False):
        if not DACGAN_AVAILABLE:
            print("  ⚠️  DA-CGAN不可用")
        else:
            print("  初始化DA-CGAN...")
            dacgan = DefectAwareConditionalGAN(device=config.get("dacgan_device", "cuda"))

            model_path = config.get("dacgan_model_path", "dacgan_pretrained.pth")
            if os.path.exists(model_path):
                dacgan.load_model(model_path)
                print(f"  ✓ 已加载DA-CGAN模型: {model_path}")
            else:
                print(f"  ⚠️  未找到模型: {model_path}")
                print("     请先训练：python main_final.py --mode train_gan")
                dacgan = None

    # ========== 处理图像 ==========
    processed_count = 0
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        image = cv2.imread(input_path)
        if image is None:
            print(f"  错误：无法加载 {input_path}")
            continue

        # ========== 处理流程 ==========

        # 步骤1：DA-CGAN增强（替换MSA）
        if config.get("use_dacgan", False) and dacgan is not None:
            original_size = image.shape[:2]
            image_resized = cv2.resize(image, (512, 512))
            image = dacgan.enhance(image_resized)
            if not (config.get("resize_width") or config.get("resize_height")):
                image = cv2.resize(image, (original_size[1], original_size[0]))

        # 步骤2：AGFPN特征提取
        if config.get("use_feature_extraction", False) and academic_extractor is not None:
            image = academic_extractor.extract_and_enhance(
                image,
                enhancement_factor=config.get("enhancement_factor", 0.05)
            )

        # 步骤3：HDE参数优化
        if config.get("use_optimization", False):
            try:
                optimal_params = HierarchicalDifferentialEvolution.hierarchical_optimize(
                    image,
                    max_generations=config.get("optimization_generations", 30)
                )
                alpha, beta, gamma, sigma, sharpen_str = optimal_params
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                image = HierarchicalDifferentialEvolution.apply_gamma(image, gamma)
                if sigma > 0.1:
                    image = cv2.GaussianBlur(image, (3, 3), sigma)
            except Exception as e:
                print(f"  优化失败: {e}")

        # 步骤4：尺寸调整
        if config.get("resize_width") or config.get("resize_height"):
            image = resize_image(image,
                                 width=config.get("resize_width"),
                                 height=config.get("resize_height"))

        # 步骤5：归一化
        if config.get("use_normalize", False):
            norm_image = normalize_image(image)
            image = (norm_image * 255).astype(np.uint8)

        cv2.imwrite(output_path, image)
        processed_count += 1

        if processed_count % 10 == 0:
            print(f"  已处理 {processed_count} 个文件...")

    print(f"  {subset}{suffix} 完成！处理了 {processed_count} 个文件")


def train_dacgan(base_dir, config):
    """训练DA-CGAN"""
    if not DACGAN_AVAILABLE:
        print("错误：未找到dacgan.py模块！")
        return

    print("\n" + "=" * 80)
    print("训练DA-CGAN")
    print("=" * 80)

    train_dir = os.path.join(base_dir, "images", "train_resized")
    print(f"\n加载训练数据: {train_dir}")

    train_images = []
    for filename in os.listdir(train_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(train_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (512, 512))
                train_images.append(img)

                if config.get("dacgan_max_train_images") and \
                        len(train_images) >= config["dacgan_max_train_images"]:
                    break

    print(f"✓ 加载了 {len(train_images)} 张训练图像")

    if len(train_images) == 0:
        print("错误：未找到训练图像！")
        return

    dacgan = DefectAwareConditionalGAN(device=config.get("dacgan_device", "cuda"))

    print("\n开始训练...")
    dacgan.train(
        train_images=train_images,
        num_epochs=config.get("dacgan_epochs", 50),
        save_interval=config.get("dacgan_save_interval", 10)
    )

    save_path = config.get("dacgan_model_path", "dacgan_pretrained.pth")
    dacgan.save_model(save_path)

    print("\n" + "=" * 80)
    print("✓ DA-CGAN训练完成！")
    print(f"  模型已保存: {save_path}")
    print("  现在运行: python main_final.py --mode process")
    print("=" * 80)


def main():
    """主程序入口"""

    parser = argparse.ArgumentParser(description='DA-CGAN + AGFPN + HDE 缺陷检测增强程序')
    parser.add_argument('--mode', type=str, default='process',
                        choices=['train_gan', 'process', 'test_gan'],
                        help='运行模式')
    parser.add_argument('--data_dir', type=str, default=r"D:\datasets\dataset\NEU-DET-YOLO",
                        help='数据集根目录')
    args = parser.parse_args()

    base_dir = args.data_dir

    # 配置
    config = {
        # === DA-CGAN ===（替换MSA）
        "use_dacgan": True,
        "dacgan_model_path": "NEU-DET_dacgan_pretrained.pth",
        "dacgan_device": "cuda",  # 或 "cpu"
        "dacgan_epochs": 50,
        "dacgan_save_interval": 500,
        "dacgan_max_train_images": None,

        # === AGFPN ===（保留）
        "use_feature_extraction": True,
        "enhancement_factor": 0.05,

        # === HDE ===（保留）
        "use_optimization": True,
        "optimization_generations": 30,

        # === 基本设置 ===
        "resize_width": 512,
        "resize_height": 512,
        "use_normalize": True,
    }

    # ========== 模式1：训练DA-CGAN ==========
    if args.mode == 'train_gan':
        train_dacgan(base_dir, config)
        return

    # ========== 模式2：测试DA-CGAN ==========
    if args.mode == 'test_gan':
        print("\n" + "=" * 80)
        print("测试DA-CGAN")
        print("=" * 80)

        if not DACGAN_AVAILABLE:
            print("错误：未找到dacgan.py模块！")
            return

        from dacgan import quick_test_dacgan
        quick_test_dacgan()
        return

    # 检查预训练模型
    if config['use_dacgan']:
        model_path = config['dacgan_model_path']
        if not os.path.exists(model_path):
            print(f"\n⚠️  警告：未找到预训练模型 {model_path}")
            print("    请先运行: python main_final.py --mode train_gan")
            print("    或设置 use_dacgan=False")

            response = input("\n是否继续（不使用DA-CGAN）？ (y/n): ")
            if response.lower() != 'y':
                return
            config['use_dacgan'] = False

    # 处理数据集
    for subset in ["train_resized", "val_resized", "test_resized"]:
        subset_path = os.path.join(base_dir, "images", subset)
        if os.path.exists(subset_path):
            print(f"\n处理 {subset} 数据集...")
            process_dataset(base_dir, subset, config, suffix="_DACGAN+HDE+AGFPN")
        else:
            print(f"警告：目录 {subset_path} 不存在")

    print("\n" + "=" * 80)
    print("处理完成！")
    print(f"\n输出目录: {os.path.join(base_dir, 'images')}")
    print("=" * 80)


if __name__ == "__main__":


    main()

