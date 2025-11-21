#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像预处理流水线
包含：灰度化、去噪、对比度增强、二值化、文本检测、透视校正、超分辨率等步骤
"""

import os
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Union
import cv2

# 尝试导入DBNet（文本检测）
DBNET_AVAILABLE = False
DBNet = None
try:
    # 尝试导入DBNet（可能有多种实现方式）
    try:
        from dbnet import DBNet
        DBNET_AVAILABLE = True
    except ImportError:
        try:
            from dbnet_pytorch import DBNet
            DBNET_AVAILABLE = True
        except ImportError:
            try:
                # 尝试从mmocr导入
                from mmocr.apis import TextDetInferencer
                DBNET_AVAILABLE = True
                DBNet = TextDetInferencer  # 使用mmocr的文本检测
            except ImportError:
                pass
except Exception as e:
    pass

if not DBNET_AVAILABLE:
    print("⚠️ DBNet未安装，文本检测功能将使用OpenCV替代。可安装: pip install dbnet-pytorch 或 pip install mmocr")

# 尝试导入Real-ESRGAN（超分辨率）
REALESRGAN_AVAILABLE = False
RealESRGANer = None
RRDBNet = None
try:
    try:
        from realesrgan import RealESRGANer
        from realesrgan.archs.rrdbnet_arch import RRDBNet
        REALESRGAN_AVAILABLE = True
    except ImportError:
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            REALESRGAN_AVAILABLE = True
        except ImportError:
            pass
except Exception as e:
    pass

if not REALESRGAN_AVAILABLE:
    print("⚠️ Real-ESRGAN未安装，超分辨率功能将使用双三次插值替代。可安装: pip install realesrgan")


class ImagePreprocessor:
    """图像预处理流水线类"""
    
    def __init__(
        self,
        use_dbnet: bool = True,
        use_realesrgan: bool = True,
        dbnet_model_path: Optional[str] = None,
        realesrgan_model_path: Optional[str] = None,
        realesrgan_scale: int = 4
    ):
        """
        初始化图像预处理器
        
        Args:
            use_dbnet: 是否使用DBNet进行文本检测
            use_realesrgan: 是否使用Real-ESRGAN进行超分辨率
            dbnet_model_path: DBNet模型路径（如果为None则尝试使用默认路径）
            realesrgan_model_path: Real-ESRGAN模型路径（如果为None则尝试下载默认模型）
            realesrgan_scale: Real-ESRGAN放大倍数（2或4）
        """
        self.use_dbnet = use_dbnet and DBNET_AVAILABLE
        self.use_realesrgan = use_realesrgan and REALESRGAN_AVAILABLE
        self.dbnet_model_path = dbnet_model_path
        self.realesrgan_model_path = realesrgan_model_path
        self.realesrgan_scale = realesrgan_scale
        
        # 初始化DBNet（如果可用）
        self.dbnet_model = None
        if self.use_dbnet and DBNET_AVAILABLE and DBNet is not None:
            try:
                if dbnet_model_path and os.path.exists(dbnet_model_path):
                    # 根据不同的DBNet实现方式初始化
                    if hasattr(DBNet, '__call__'):
                        self.dbnet_model = DBNet(model_path=dbnet_model_path)
                    else:
                        self.dbnet_model = DBNet(model=dbnet_model_path)
                else:
                    # 尝试使用默认模型
                    if hasattr(DBNet, '__call__'):
                        self.dbnet_model = DBNet()
                    else:
                        # 对于mmocr，需要指定模型名称
                        self.dbnet_model = DBNet(model='dbnet')
                print("✅ DBNet模型加载成功")
            except Exception as e:
                print(f"⚠️ DBNet模型加载失败: {e}")
                self.use_dbnet = False
                self.dbnet_model = None
        elif self.use_dbnet:
            print("⚠️ DBNet不可用，将使用OpenCV进行文本区域检测")
            self.use_dbnet = False
        
        # 初始化Real-ESRGAN（如果可用）
        self.realesrgan_upsampler = None
        if self.use_realesrgan and REALESRGAN_AVAILABLE and RealESRGANer is not None and RRDBNet is not None:
            try:
                if realesrgan_model_path and os.path.exists(realesrgan_model_path):
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                   num_block=23, num_grow_ch=32, scale=realesrgan_scale)
                    self.realesrgan_upsampler = RealESRGANer(
                        scale=realesrgan_scale,
                        model_path=realesrgan_model_path,
                        model=model,
                        tile=0,
                        tile_pad=10,
                        pre_pad=0,
                        half=False
                    )
                else:
                    # 尝试使用默认模型
                    if realesrgan_scale == 2:
                        model_name = 'realesrgan-x2plus.pth'
                    else:
                        model_name = 'realesrgan-x4plus.pth'
                    
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                   num_block=23, num_grow_ch=32, scale=realesrgan_scale)
                    self.realesrgan_upsampler = RealESRGANer(
                        scale=realesrgan_scale,
                        model_path=f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}',
                        model=model,
                        tile=0,
                        tile_pad=10,
                        pre_pad=0,
                        half=False
                    )
                print(f"✅ Real-ESRGAN模型加载成功（放大倍数: {realesrgan_scale}x）")
            except Exception as e:
                print(f"⚠️ Real-ESRGAN模型加载失败: {e}")
                self.use_realesrgan = False
                self.realesrgan_upsampler = None
        elif self.use_realesrgan:
            print("⚠️ Real-ESRGAN不可用，将使用双三次插值进行超分辨率")
            self.use_realesrgan = False
    
    def load_image(self, image_path: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        加载图像
        
        Args:
            image_path: 图像路径、PIL Image对象或numpy数组
            
        Returns:
            BGR格式的numpy数组
        """
        if isinstance(image_path, np.ndarray):
            # 如果已经是numpy数组，直接返回
            if len(image_path.shape) == 2:
                # 灰度图转BGR
                return cv2.cvtColor(image_path, cv2.COLOR_GRAY2BGR)
            elif len(image_path.shape) == 3:
                # 如果是RGB，转BGR
                if image_path.shape[2] == 3:
                    return cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
                return image_path
            return image_path
        elif isinstance(image_path, Image.Image):
            # PIL Image转numpy数组
            img_array = np.array(image_path)
            if len(img_array.shape) == 2:
                return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif len(img_array.shape) == 3:
                if img_array.shape[2] == 3:
                    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_array
        else:
            # 从文件路径加载
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法加载图像: {image_path}")
            return img
    
    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        灰度化
        
        Args:
            image: BGR格式的图像
            
        Returns:
            灰度图像
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def denoise(self, image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        双边滤波去噪
        
        Args:
            image: 灰度图像
            d: 滤波时每个像素邻域的直径
            sigma_color: 颜色空间的标准差
            sigma_space: 坐标空间的标准差
            
        Returns:
            去噪后的图像
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def enhance_contrast(self, image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        使用CLAHE增强对比度
        
        Args:
            image: 灰度图像
            clip_limit: 对比度限制
            tile_grid_size: 网格大小
            
        Returns:
            对比度增强后的图像
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    def adaptive_threshold(self, image: np.ndarray, max_value: int = 255, 
                          adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          threshold_type: int = cv2.THRESH_BINARY,
                          block_size: int = 11, C: float = 2) -> np.ndarray:
        """
        自适应阈值二值化
        
        Args:
            image: 灰度图像
            max_value: 二值化后的最大值
            adaptive_method: 自适应方法
            threshold_type: 阈值类型
            block_size: 邻域大小（必须是奇数）
            C: 从均值或加权和中减去的常数
            
        Returns:
            二值化图像
        """
        # 确保block_size是奇数
        if block_size % 2 == 0:
            block_size += 1
        return cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, C)
    
    def detect_text_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """
        使用DBNet检测文本区域（如果不可用，则使用OpenCV进行简单检测）
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            文本区域列表（每个区域是一个numpy数组）
        """
        if not self.use_dbnet or self.dbnet_model is None:
            # 如果DBNet不可用，使用OpenCV进行简单的文本区域检测
            return self._detect_text_regions_opencv(image)
        
        try:
            # 使用DBNet进行文本检测
            boxes = None
            
            # 尝试不同的DBNet API
            if hasattr(self.dbnet_model, 'detect'):
                boxes = self.dbnet_model.detect(image)
            elif hasattr(self.dbnet_model, '__call__'):
                # mmocr的TextDetInferencer
                result = self.dbnet_model(image)
                if isinstance(result, dict) and 'predictions' in result:
                    boxes = result['predictions']
                elif isinstance(result, list):
                    boxes = result
            else:
                # 尝试直接调用
                result = self.dbnet_model(image)
                boxes = result if isinstance(result, (list, np.ndarray)) else None
            
            if boxes is None or len(boxes) == 0:
                # 如果没有检测到文本区域，使用OpenCV方法
                return self._detect_text_regions_opencv(image)
            
            # 提取文本区域
            text_regions = []
            for box in boxes:
                # 处理不同格式的检测框
                if isinstance(box, np.ndarray):
                    if box.shape == (4, 2):
                        # 四个角点格式
                        x_coords = box[:, 0]
                        y_coords = box[:, 1]
                        x1, y1 = int(x_coords.min()), int(y_coords.min())
                        x2, y2 = int(x_coords.max()), int(y_coords.max())
                    elif box.shape == (4,):
                        # [x1, y1, x2, y2]格式
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    else:
                        continue
                elif isinstance(box, (list, tuple)) and len(box) >= 4:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                else:
                    continue
                
                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    text_regions.append(image[y1:y2, x1:x2])
            
            if not text_regions:
                # 如果没有检测到文本区域，使用OpenCV方法
                return self._detect_text_regions_opencv(image)
            
            return text_regions
        except Exception as e:
            print(f"⚠️ DBNet文本检测失败: {e}，使用OpenCV替代方法")
            return self._detect_text_regions_opencv(image)
    
    def _detect_text_regions_opencv(self, image: np.ndarray) -> List[np.ndarray]:
        """
        使用OpenCV进行简单的文本区域检测（DBNet的替代方案）
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            文本区域列表
        """
        gray = self.grayscale(image) if len(image.shape) == 3 else image
        
        # 使用形态学操作检测文本区域
        # 创建水平核用于检测水平文本
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detected_lines = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # 过滤太小的区域
            if w > 50 and h > 10:
                # 添加一些边距
                margin = 5
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image.shape[1], x + w + margin)
                y2 = min(image.shape[0], y + h + margin)
                text_regions.append(image[y1:y2, x1:x2])
        
        if not text_regions:
            # 如果没有检测到文本区域，返回整个图像
            return [image]
        
        return text_regions
    
    def perspective_correction(self, image: np.ndarray, corners: Optional[np.ndarray] = None) -> np.ndarray:
        """
        透视校正
        
        Args:
            image: 输入图像
            corners: 四个角点坐标（如果为None，则尝试自动检测）
            
        Returns:
            校正后的图像
        """
        if corners is None:
            # 尝试自动检测角点（使用轮廓检测）
            gray = self.grayscale(image) if len(image.shape) == 3 else image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                # 近似为多边形
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                if len(approx) == 4:
                    corners = approx.reshape(4, 2)
                else:
                    # 如果无法找到4个角点，返回原图
                    return image
            else:
                return image
        
        # 定义目标矩形的四个角点
        h, w = image.shape[:2]
        dst_corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
        
        # 应用透视变换
        corrected = cv2.warpPerspective(image, M, (w, h))
        return corrected
    
    def rotation_correction(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """
        旋转校正
        
        Args:
            image: 输入图像
            angle: 旋转角度（如果为None，则尝试自动检测）
            
        Returns:
            校正后的图像
        """
        if angle is None:
            # 尝试自动检测旋转角度（使用Hough变换检测直线）
            gray = self.grayscale(image) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            
            if lines is not None and len(lines) > 0:
                # 计算平均角度
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle_deg = np.degrees(theta) - 90
                    if abs(angle_deg) < 45:  # 只考虑接近水平的线
                        angles.append(angle_deg)
                
                if angles:
                    angle = np.median(angles)
                else:
                    return image
            else:
                return image
        
        # 旋转图像
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def crop_and_correct(self, image: np.ndarray) -> np.ndarray:
        """
        图像裁剪 + 透视校正 + 旋转校正
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        # 先进行旋转校正
        corrected = self.rotation_correction(image)
        
        # 再进行透视校正
        corrected = self.perspective_correction(corrected)
        
        return corrected
    
    def super_resolution(self, image: np.ndarray) -> np.ndarray:
        """
        使用Real-ESRGAN进行超分辨率放大
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            放大后的图像
        """
        if not self.use_realesrgan or self.realesrgan_upsampler is None:
            # 如果Real-ESRGAN不可用，使用双三次插值放大
            h, w = image.shape[:2]
            upscaled = cv2.resize(image, (w * self.realesrgan_scale, h * self.realesrgan_scale), 
                                interpolation=cv2.INTER_CUBIC)
            return upscaled
        
        try:
            # Real-ESRGAN需要RGB格式
            if len(image.shape) == 3 and image.shape[2] == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image
            
            # 执行超分辨率
            output, _ = self.realesrgan_upsampler.enhance(img_rgb, outscale=self.realesrgan_scale)
            
            # 转回BGR格式
            if len(output.shape) == 3:
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            return output
        except Exception as e:
            print(f"⚠️ 超分辨率处理失败: {e}，使用双三次插值代替")
            h, w = image.shape[:2]
            return cv2.resize(image, (w * self.realesrgan_scale, h * self.realesrgan_scale), 
                            interpolation=cv2.INTER_CUBIC)
    
    def process_pipeline(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        save_intermediate: bool = False,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        完整的图像预处理流水线
        
        Args:
            image_input: 输入图像（路径、PIL Image或numpy数组）
            save_intermediate: 是否保存中间结果
            output_path: 输出路径（如果为None则不保存）
            
        Returns:
            处理后的图像（BGR格式）
        """
        print("=" * 60)
        print("开始图像预处理流水线")
        print("=" * 60)
        
        # 1. 图像输入
        print("1. 加载图像...")
        image = self.load_image(image_input)
        print(f"   图像尺寸: {image.shape}")
        
        if save_intermediate:
            cv2.imwrite("step1_input.png", image)
        
        # 2. 灰度化
        print("2. 灰度化...")
        gray = self.grayscale(image)
        if save_intermediate:
            cv2.imwrite("step2_grayscale.png", gray)
        
        # 3. 去噪（双边滤波）
        print("3. 去噪（双边滤波）...")
        denoised = self.denoise(gray)
        if save_intermediate:
            cv2.imwrite("step3_denoised.png", denoised)
        
        # 4. 增强对比度（CLAHE）
        print("4. 增强对比度（CLAHE）...")
        enhanced = self.enhance_contrast(denoised)
        if save_intermediate:
            cv2.imwrite("step4_enhanced.png", enhanced)
        
        # 5. 二值化（Adaptive Threshold）
        print("5. 二值化（Adaptive Threshold）...")
        binary = self.adaptive_threshold(enhanced)
        if save_intermediate:
            cv2.imwrite("step5_binary.png", binary)
        
        # 6. 跳过文本区域检测和裁剪，直接对整个图像进行超分辨率放大
        print("6. 对整个图像进行超分辨率放大...")
        # 使用预处理后的图像进行超分辨率
        # 如果是彩色图像，先对每个通道进行CLAHE增强，然后进行超分辨率
        if len(image.shape) == 3:
            # 彩色图像：对每个通道分别进行CLAHE增强
            enhanced_color = np.zeros_like(image)
            for i in range(3):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_color[:, :, i] = clahe.apply(image[:, :, i])
            image_for_sr = enhanced_color
        else:
            # 灰度图，使用增强后的图像转换为BGR格式
            image_for_sr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        final_image = self.super_resolution(image_for_sr)
        if save_intermediate:
            cv2.imwrite("step6_upscaled.png", final_image)
        
        # 保存最终结果
        if output_path:
            cv2.imwrite(output_path, final_image)
            print(f"✅ 最终结果已保存到: {output_path}")
        
        print("=" * 60)
        print("图像预处理流水线完成")
        print("=" * 60)
        
        return final_image


def main():
    """示例使用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="图像预处理流水线")
    parser.add_argument("input", help="输入图像路径")
    parser.add_argument("-o", "--output", help="输出图像路径", default="output.png")
    parser.add_argument("--save-intermediate", action="store_true", help="保存中间处理结果")
    parser.add_argument("--no-dbnet", action="store_true", help="不使用DBNet")
    parser.add_argument("--no-realesrgan", action="store_true", help="不使用Real-ESRGAN")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4], help="超分辨率放大倍数")
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = ImagePreprocessor(
        use_dbnet=not args.no_dbnet,
        use_realesrgan=not args.no_realesrgan,
        realesrgan_scale=args.scale
    )
    
    # 执行预处理流水线
    result = preprocessor.process_pipeline(
        args.input,
        save_intermediate=args.save_intermediate,
        output_path=args.output
    )
    
    print(f"\n处理完成！输出图像尺寸: {result.shape}")


if __name__ == "__main__":
    main()

