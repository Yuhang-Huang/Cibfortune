#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级图像去模糊和清晰化处理
包含：维纳滤波、Richardson-Lucy去卷积、清晰度评估、多尺度处理等
"""

import cv2
import numpy as np
import os

# scipy是可选的，只在需要时导入
try:
    from scipy import ndimage, signal
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 尝试导入PaddleOCR
PADDLEOCR_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except Exception as e:
    # PaddleOCR导入失败，但不影响其他功能
    pass




def evaluate_clarity(image):
    """
    评估图像清晰度（使用Laplacian方差）
    
    Args:
        image: 输入图像（灰度图）
    
    Returns:
        清晰度分数（方差越大越清晰）
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    mean, stddev = cv2.meanStdDev(laplacian)
    clarity = stddev[0][0] * stddev[0][0]  # 方差
    return clarity


def create_motion_blur_kernel(length, angle):
    """
    创建运动模糊核（点扩散函数PSF）
    
    Args:
        length: 模糊长度（像素）
        angle: 模糊角度（度）
    
    Returns:
        模糊核
    """
    kernel = np.zeros((length, length), dtype=np.float32)
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    center = length // 2
    for i in range(length):
        x = int(center + (i - center) * cos_angle)
        y = int(center + (i - center) * sin_angle)
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    
    kernel = kernel / np.sum(kernel)
    return kernel


def wiener_filter_deblur(image, kernel_size=15, sigma=5.0, noise_var=0.01):
    """
    维纳滤波去模糊
    
    Args:
        image: 输入模糊图像（灰度图）
        kernel_size: 模糊核大小
        sigma: 高斯核标准差
        noise_var: 噪声方差
    
    Returns:
        去模糊后的图像
    """
    # 创建高斯模糊核
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel @ kernel.T
    kernel = kernel / np.sum(kernel)
    
    # 转换到频域
    image_float = image.astype(np.float32) / 255.0
    h, w = image_float.shape
    
    # 扩展图像和核到相同大小（避免边界效应）
    pad_h, pad_w = h + kernel_size, w + kernel_size
    image_padded = np.zeros((pad_h, pad_w), dtype=np.float32)
    kernel_padded = np.zeros((pad_h, pad_w), dtype=np.float32)
    
    image_padded[:h, :w] = image_float
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel
    
    # FFT
    image_fft = np.fft.fft2(image_padded)
    kernel_fft = np.fft.fft2(np.fft.fftshift(kernel_padded))
    
    # 维纳滤波公式: H*(u,v) / (|H(u,v)|^2 + K)
    # 其中K = noise_var / signal_var
    kernel_mag_sq = np.abs(kernel_fft) ** 2
    K = noise_var / (1.0 - noise_var)  # 简化的噪声比
    wiener_filter = np.conj(kernel_fft) / (kernel_mag_sq + K)
    
    # 应用滤波器
    deblurred_fft = image_fft * wiener_filter
    deblurred = np.real(np.fft.ifft2(deblurred_fft))
    
    # 裁剪回原始大小
    deblurred = deblurred[:h, :w]
    
    # 归一化到0-255
    deblurred = np.clip(deblurred * 255.0, 0, 255).astype(np.uint8)
    
    return deblurred


def richardson_lucy_deconv(image, psf, iterations=30):
    """
    Richardson-Lucy去卷积算法
    
    Args:
        image: 输入模糊图像（灰度图）
        psf: 点扩散函数（模糊核）
        iterations: 迭代次数
    
    Returns:
        恢复后的图像
    """
    # 转换为浮点数
    image_float = image.astype(np.float64) / 255.0
    psf_float = psf.astype(np.float64)
    psf_float = psf_float / np.sum(psf_float)
    
    # 初始化估计
    estimate = np.full_like(image_float, 0.5)
    
    # 翻转PSF（用于卷积）
    psf_flipped = np.flip(np.flip(psf_float, 0), 1)
    
    for i in range(iterations):
        # 前向卷积：估计 * PSF
        blurred_estimate = cv2.filter2D(estimate, -1, psf_float, borderType=cv2.BORDER_REFLECT)
        
        # 避免除零
        blurred_estimate = np.clip(blurred_estimate, 1e-10, 1.0)
        
        # 计算比率
        ratio = image_float / blurred_estimate
        
        # 反向卷积：比率 * 翻转PSF
        correction = cv2.filter2D(ratio, -1, psf_flipped, borderType=cv2.BORDER_REFLECT)
        
        # 更新估计
        estimate = estimate * correction
        
        # 确保值在有效范围内
        estimate = np.clip(estimate, 0, 1)
    
    # 转换回0-255
    result = np.clip(estimate * 255.0, 0, 255).astype(np.uint8)
    return result


def enhance_image_preprocessing(image):
    """
    完整的图像预处理流程（用于OCR优化）
    
    Args:
        image: 输入图像（灰度图）
    
    Returns:
        预处理后的图像
    """
    # 1. CLAHE对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    # 2. 二值化（OTSU自适应阈值）
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. 形态学操作（闭运算，连接断开的字符）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return processed


def multi_scale_processing(image, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
    """
    多尺度处理
    
    Args:
        image: 输入图像
        scales: 尺度列表
    
    Returns:
        处理后的图像列表
    """
    results = []
    for scale in scales:
        h, w = image.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        results.append((scale, resized))
    return results


def paddleocr_super_resolution(image_path, output_path="paddleocr_sr_output.jpg", 
                               scale=None, target_width=None, target_height=None, 
                               max_width=1200, max_height=1200):
    """
    使用双三次插值进行图像超分辨率
    
    默认限制最大尺寸为2500x2500像素，既能保证清晰度又不会影响OCR识别速度。
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像保存路径
        scale: 放大倍数（如果指定了target_width/target_height或max_width/max_height则忽略）
        target_width: 目标宽度（像素），如果指定则按此宽度缩放，高度按比例调整
        target_height: 目标高度（像素），如果指定则按此高度缩放，宽度按比例调整
        max_width: 最大宽度（像素），默认2500，如果超过则按比例缩小
        max_height: 最大高度（像素），默认2500，如果超过则按比例缩小
    
    Returns:
        放大后的图像数组，如果失败返回None
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在: {image_path}")
        return None
    
    print(f"\n=== 使用双三次插值进行图像超分辨率 ===")
    print(f"正在读取图像: {image_path}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        try:
            from PIL import Image
            pil_img = Image.open(image_path).convert("RGB")
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"无法读取图像: {e}")
            return None
    
    if image is None or image.size == 0:
        print("错误: 无法读取图像或图像为空")
        return None
    
    original_shape = image.shape
    h, w = image.shape[:2]
    print(f"原始图像尺寸: {w}x{h}")
    
    try:
        # 计算目标尺寸
        new_w = w
        new_h = h
        
        # 优先级1: 如果指定了目标宽度或高度
        if target_width is not None or target_height is not None:
            if target_width is not None and target_height is not None:
                # 同时指定宽度和高度，直接使用
                new_w = target_width
                new_h = target_height
                print(f"指定目标尺寸: {new_w}x{new_h}")
            elif target_width is not None:
                # 只指定宽度，高度按比例
                new_w = target_width
                new_h = int(h * (target_width / w))
                print(f"指定目标宽度: {new_w}，高度按比例: {new_h}")
            elif target_height is not None:
                # 只指定高度，宽度按比例
                new_h = target_height
                new_w = int(w * (target_height / h))
                print(f"指定目标高度: {new_h}，宽度按比例: {new_w}")
        
        # 优先级2: 使用scale参数
        elif scale is not None:
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"使用放大倍数: {scale}x，目标尺寸: {new_w}x{new_h}")
        
        # 优先级3: 应用默认最大尺寸限制
        else:
            # 默认情况下，如果图像超过最大尺寸，则按比例缩小
            # 如果图像小于最大尺寸，适当放大（但不超过最大尺寸，最多放大2倍避免模糊）
            scale_w = max_width / w if w > max_width else min(max_width / w, 2.0)
            scale_h = max_height / h if h > max_height else min(max_height / h, 2.0)
            final_scale = min(scale_w, scale_h)
            
            new_w = int(w * final_scale)
            new_h = int(h * final_scale)
            
            if w > max_width or h > max_height:
                print(f"应用默认最大尺寸限制: {max_width}x{max_height}，缩放后: {new_w}x{new_h}")
            elif final_scale > 1.0:
                print(f"图像较小，适当放大至: {new_w}x{new_h}（不超过最大尺寸限制: {max_width}x{max_height}）")
            else:
                print(f"图像尺寸合适，保持原始尺寸: {w}x{h}")
        
        # 最后检查：确保不超过最大尺寸限制（防止其他优先级计算出的尺寸超过限制）
        if new_w > max_width or new_h > max_height:
            scale_w = max_width / new_w if new_w > max_width else 1.0
            scale_h = max_height / new_h if new_h > max_height else 1.0
            final_scale = min(scale_w, scale_h)
            
            new_w = int(new_w * final_scale)
            new_h = int(new_h * final_scale)
            print(f"最终应用最大尺寸限制: {max_width}x{max_height}，调整后: {new_w}x{new_h}")
        
        # 使用双三次插值进行缩放
        print(f"正在使用双三次插值缩放图像...")
        upscaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        final_shape = upscaled_image.shape
        actual_scale_w = new_w / w
        actual_scale_h = new_h / h
        print(f"缩放后图像尺寸: {final_shape[1]}x{final_shape[0]}")
        print(f"宽度缩放: {actual_scale_w:.2f}x, 高度缩放: {actual_scale_h:.2f}x")
        
        # 评估清晰度
        gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_upscaled = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
        clarity_original = evaluate_clarity(gray_original)
        clarity_upscaled = evaluate_clarity(gray_upscaled)
        
        print(f"   原始清晰度: {clarity_original:.2f}")
        print(f"   放大后清晰度: {clarity_upscaled:.2f}")
        print(f"   清晰度变化: {clarity_upscaled/clarity_original:.2f}x")
        
        # 保存图像
        print(f"正在保存图像到: {output_path}")
        cv2.imwrite(output_path, upscaled_image, [cv2.IMWRITE_JPEG_QUALITY, 98])
        
        print(f"\n处理完成！")
        print(f"   输出文件: {output_path}")
        
        return upscaled_image
        
    except Exception as e:
        error_msg = f"处理失败: {e}"
        print(f"错误: {error_msg}")
        import traceback
        traceback.print_exc()
        raise Exception(error_msg)


def blur_to_sharp(image_path, output_path="sharpened_output.jpg", method='auto'):
    """
    高级图像去模糊和清晰化处理
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        method: 处理方法
            - 'auto': 自动选择最佳方法
            - 'wiener': 维纳滤波
            - 'richardson_lucy': Richardson-Lucy去卷积
            - 'unsharp': Unsharp Masking锐化
            - 'combined': 组合方法
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在: {image_path}")
        return None
    
    # 读取图像
    print(f"正在读取图像: {image_path}")
    image = cv2.imread(image_path)
    
    # 如果cv2.imread失败，尝试用其他方法
    if image is None:
        print("尝试使用其他方法读取图像...")
        try:
            from PIL import Image
            pil_img = Image.open(image_path).convert("RGB")
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            print("使用PIL成功读取图像")
        except Exception as e:
            print(f"无法读取图像: {e}")
            return None
    
    if image is None or image.size == 0:
        print("错误: 无法读取图像或图像为空")
        return None
    
    print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 评估原始清晰度
    original_clarity = evaluate_clarity(gray)
    print(f"原始图像清晰度: {original_clarity:.2f}")
    
    results = {}
    
    # 方法1: Unsharp Masking（快速，适合轻微模糊）
    if method in ['auto', 'unsharp', 'combined']:
        print("\n=== 方法1: Unsharp Masking锐化 ===")
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        blurred2 = cv2.GaussianBlur(blurred, (0, 0), 3)
        unsharp_result = cv2.addWeighted(gray, 1.5, blurred2, -0.5, 0)
        unsharp_result = np.clip(unsharp_result, 0, 255).astype(np.uint8)
        clarity_unsharp = evaluate_clarity(unsharp_result)
        results['unsharp'] = (unsharp_result, clarity_unsharp)
        print(f"清晰度: {clarity_unsharp:.2f} (提升: {clarity_unsharp/original_clarity:.2f}x)")
    
    # 方法2: 维纳滤波去模糊
    if method in ['auto', 'wiener', 'combined']:
        print("\n=== 方法2: 维纳滤波去模糊 ===")
        try:
            wiener_result = wiener_filter_deblur(gray, kernel_size=15, sigma=5.0, noise_var=0.01)
            clarity_wiener = evaluate_clarity(wiener_result)
            results['wiener'] = (wiener_result, clarity_wiener)
            print(f"清晰度: {clarity_wiener:.2f} (提升: {clarity_wiener/original_clarity:.2f}x)")
        except Exception as e:
            print(f"维纳滤波处理失败: {e}")
    
    # 方法3: Richardson-Lucy去卷积
    if method in ['auto', 'richardson_lucy', 'combined']:
        print("\n=== 方法3: Richardson-Lucy去卷积 ===")
        try:
            # 创建运动模糊核（假设轻微运动模糊）
            psf = create_motion_blur_kernel(length=15, angle=45)
            rl_result = richardson_lucy_deconv(gray, psf, iterations=30)
            clarity_rl = evaluate_clarity(rl_result)
            results['richardson_lucy'] = (rl_result, clarity_rl)
            print(f"清晰度: {clarity_rl:.2f} (提升: {clarity_rl/original_clarity:.2f}x)")
        except Exception as e:
            print(f"Richardson-Lucy处理失败: {e}")
    
    # 选择最佳结果
    if method == 'auto' and results:
        best_method = max(results.items(), key=lambda x: x[1][1])
        print(f"\n=== 自动选择最佳方法: {best_method[0]} ===")
        final_result = best_method[1][0]
    elif method in results:
        final_result = results[method][0]
    elif 'unsharp' in results:
        final_result = results['unsharp'][0]
    else:
        print("错误: 没有可用的处理方法")
        return None
    
    # 后处理：预处理优化（用于OCR）
    print("\n=== 后处理: OCR优化预处理 ===")
    processed = enhance_image_preprocessing(final_result)
    
    # 保存结果
    print(f"\n正在保存结果到: {output_path}")
    cv2.imwrite(output_path, final_result, [cv2.IMWRITE_JPEG_QUALITY, 98])
    
    # 保存预处理版本（用于OCR）
    preprocessed_path = output_path.replace('.jpg', '_preprocessed.jpg').replace('.png', '_preprocessed.png')
    cv2.imwrite(preprocessed_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 98])
    
    final_clarity = evaluate_clarity(final_result)
    print(f"\n处理完成！")
    print(f"   原始清晰度: {original_clarity:.2f}")
    print(f"   最终清晰度: {final_clarity:.2f}")
    print(f"   清晰度提升: {final_clarity/original_clarity:.2f}x")
    print(f"   输出文件: {output_path}")
    print(f"   预处理文件: {preprocessed_path}")
    
    # 可选：显示图像（如果有GUI环境）
    try:
        cv2.imshow("Original", gray)
        cv2.imshow("Enhanced", final_result)
        cv2.imshow("Preprocessed (for OCR)", processed)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("无法显示图像（可能没有GUI环境），但已保存结果")
    
    return final_result


if __name__ == "__main__":
    # 使用相对路径或正确的路径格式
    # 方法1: 使用原始字符串（推荐）
    image_path = r"ticket.webp"
    
    # 方法2: 如果文件在其他位置，使用正斜杠
    # image_path = "D:/cibfortune/Cibfortune/cibfortune/ticket.webp"
    
    # 如果文件在特定位置，也可以这样：
    # import os
    # image_path = os.path.join("D:", "cibfortune", "Cibfortune", "cibfortune", "ticket.webp")
    
    # ========== 功能选择 ==========
    
    # 1. 传统图像去模糊和清晰化
    # 方法选项：
    # - 'auto': 自动选择最佳方法（推荐）
    # - 'wiener': 维纳滤波去模糊
    # - 'richardson_lucy': Richardson-Lucy去卷积
    # - 'unsharp': Unsharp Masking锐化
    # - 'combined': 组合所有方法
    blur_to_sharp(image_path, output_path="sharpened_output.jpg", method='auto')
    
    # 2. 使用双三次插值进行图像超分辨率
    # 默认会自动应用最大尺寸限制（2500x2500），既能保证清晰度又不会影响OCR识别速度
    # 
    # 方式1: 使用默认设置（推荐，自动限制最大尺寸为2500x2500）
    # paddleocr_super_resolution(
    #     image_path, 
    #     output_path="paddleocr_sr_output.jpg"
    # )
    #
    # 方式2: 使用放大倍数（仍会受最大尺寸限制）
    # paddleocr_super_resolution(
    #     image_path, 
    #     output_path="paddleocr_sr_output.jpg",
    #     scale=4  # 放大4倍，但不超过2500x2500
    # )
    #
    # 方式3: 指定目标宽度（高度按比例）
    # paddleocr_super_resolution(
    #     image_path,
    #     output_path="paddleocr_sr_output.jpg",
    #     target_width=2000  # 目标宽度2000像素
    # )
    #
    # 方式4: 自定义最大尺寸限制
    # paddleocr_super_resolution(
    #     image_path,
    #     output_path="paddleocr_sr_output.jpg",
    #     max_width=3000,  # 自定义最大宽度3000像素
    #     max_height=3000  # 自定义最大高度3000像素
    # )