# -*- coding: utf-8 -*-
"""
印章消除模块
通过红色通道处理和色阶调整来消除图片中的红色印章，保留文字内容
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import os


class SealRemover:
    """印章消除器"""
    
    def __init__(self):
        """初始化印章消除器"""
        pass
    
    def remove_seal(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        gamma: float = 2.0,
        save_intermediate: bool = True,
        intermediate_dir: Optional[str] = None
    ) -> np.ndarray:
        """
        消除图片中的红色印章（仅通过色阶调整，不使用掩码）
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径，如果为None则不保存
            gamma: 色阶调整的gamma值（必须 > 1），值越大，红色印章越接近白色，文字越接近黑色
                  建议范围：1.5-3.0，默认2.0
                  - gamma > 1 时：低值被压缩（文字变黑），高值被拉伸（印章变白）
            save_intermediate: 是否保存中间处理步骤的图片
            intermediate_dir: 中间图片保存目录，如果为None则使用output_path的目录
            
        Returns:
            处理后的图像（BGR格式）
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 确定中间图片保存目录
        if save_intermediate:
            if intermediate_dir is None:
                if output_path:
                    intermediate_dir = os.path.dirname(output_path) or "."
                else:
                    intermediate_dir = "."
            os.makedirs(intermediate_dir, exist_ok=True)
            
            # 获取输入文件名（不含扩展名）用于命名中间文件
            base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 分离BGR通道
        b, g, r = cv2.split(img)
        
        # 转换为float32以便进行计算
        b = b.astype(np.float32)
        g = g.astype(np.float32)
        r = r.astype(np.float32)
        
        # 保存步骤1：原始红色通道
        if save_intermediate:
            red_channel_path = os.path.join(intermediate_dir, f"{base_name}_step1_red_channel.jpg")
            cv2.imwrite(red_channel_path, r.astype(np.uint8))
            print(f"步骤1 - 原始红色通道已保存: {red_channel_path}")
        
        # 1. 检测红色区域 - 使用HSV颜色空间更准确
        # 转换为HSV颜色空间进行红色检测
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 红色在HSV中的范围（红色在色环两端：0-10 和 160-180）
        # 放宽范围以检测更多红色区域
        red_mask_hsv1 = (h >= 0) & (h <= 15)  # 扩大范围到15
        red_mask_hsv2 = (h >= 160) & (h <= 180)
        red_mask_hsv = red_mask_hsv1 | red_mask_hsv2
        
        # 高饱和度区域（印章通常饱和度较高，但也要考虑低饱和度的情况）
        high_sat_mask = s > 50  # 降低饱和度阈值，检测更多区域
        
        # 中等以上明度（避免检测到太暗的区域）
        medium_v_mask = v > 50  # 降低明度阈值
        
        # HSV检测：红色且有一定饱和度和明度
        red_mask_hsv_final = red_mask_hsv & high_sat_mask & medium_v_mask
        
        # 同时使用RGB空间检测作为补充（放宽条件）
        red_mask_rgb = (r > 120) & (g < 120) & (b < 120) & (r > g + 30) & (r > b + 30)
        
        # 合并两种检测方法的结果
        red_mask = red_mask_hsv_final | red_mask_rgb
        
        # 先显示图像中的RGB值范围，帮助调试
        if save_intermediate:
            print(f"图像RGB值范围: R=[{r.min():.1f}, {r.max():.1f}], G=[{g.min():.1f}, {g.max():.1f}], B=[{b.min():.1f}, {b.max():.1f}]")
            print(f"HSV检测到的红色像素: {np.sum(red_mask_hsv_final)}")
            print(f"RGB检测到的红色像素: {np.sum(red_mask_rgb)}")
            print(f"合并后检测到的红色像素: {np.sum(red_mask)}")
        
        # 检查是否检测到印章区域
        if not np.any(red_mask):
            print("警告: 未检测到红色印章区域，将返回原始图像")
            print("提示: 如果图像中有红色印章但未检测到，可能需要进一步调整检测阈值")
            if output_path:
                cv2.imwrite(output_path, img)
                print(f"原始图像已保存到: {output_path}")
            return img
        
        # 对检测到的红色区域进行形态学操作，填充小空洞和连接断开的区域
        red_mask_uint8 = (red_mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # 先闭运算（填充小空洞）
        red_mask_uint8 = cv2.morphologyEx(red_mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        # 再开运算（去除小噪点）
        red_mask_uint8 = cv2.morphologyEx(red_mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
        # 膨胀操作（扩大检测区域，确保边缘也被检测到）
        red_mask_uint8 = cv2.dilate(red_mask_uint8, kernel, iterations=1)
        # 转换回布尔掩码
        red_mask = red_mask_uint8 > 0
        
        if save_intermediate:
            print(f"形态学操作后检测到的红色像素: {np.sum(red_mask)}")
        
        # 验证gamma参数
        if gamma <= 0:
            print(f"警告: gamma值必须 > 0，当前值为 {gamma}，使用默认值 2.0")
            gamma = 2.0
        elif gamma <= 1:
            print(f"警告: gamma值应该 > 1 才能有效降低红色，当前值为 {gamma}，使用默认值 2.0")
            gamma = 2.0
        
        # 保存步骤2：红色区域掩码
        if save_intermediate:
            red_mask_vis = (red_mask.astype(np.uint8) * 255)
            mask_path = os.path.join(intermediate_dir, f"{base_name}_step2_red_mask.jpg")
            cv2.imwrite(mask_path, red_mask_vis)
            print(f"步骤2 - 红色区域掩码已保存: {mask_path}")
            print(f"检测到的印章像素数量: {np.sum(red_mask)}")
        
        # 2. 在红色区域进行色阶调整，使红色印章趋近白色
        # 方法：对红色通道应用gamma校正（降低红色）
        #       对绿色和蓝色通道进行提升（提高绿色和蓝色）
        
        # 归一化到0-1范围
        r_normalized = r / 255.0
        g_normalized = g / 255.0
        b_normalized = b / 255.0
        
        # 初始化调整后的通道，非印章区域保持原值
        r_adjusted = r_normalized.copy()
        g_adjusted = g_normalized.copy()
        b_adjusted = b_normalized.copy()
        
        # 只在印章区域进行调整
        # 对红色通道应用gamma校正（gamma > 1，降低红色值）
        if np.any(red_mask):
            # 获取印章区域的原始红色值（用于调试）
            original_r_values = r_normalized[red_mask]
            
            # 对红色通道应用gamma校正（gamma > 1，降低红色值）
            r_adjusted[red_mask] = np.power(r_normalized[red_mask], gamma)
            
            # 调试信息：显示调整前后的值
            if save_intermediate:
                print(f"红色通道调整: 原始值范围 [{original_r_values.min():.3f}, {original_r_values.max():.3f}]")
                print(f"红色通道调整: gamma校正后范围 [{r_adjusted[red_mask].min():.3f}, {r_adjusted[red_mask].max():.3f}]")
                print(f"使用的gamma值: {gamma}")
            
            # 在印章区域，提高绿色和蓝色通道，使RGB三个通道接近（形成白色）
            original_g_values = g_normalized[red_mask]
            original_b_values = b_normalized[red_mask]
            
            # 策略：设定一个固定的高目标白色值（0.98），让RGB三个通道都向这个值靠拢
            # 这样能确保形成真正的白色，而不是灰绿色
            target_white = 0.98  # 目标白色值（0.98 = 约250/255，非常白）
            
            # 对于绿色和蓝色通道，直接提升到目标值附近
            inv_gamma = 1.0 / gamma
            g_adjusted[red_mask] = np.clip(
                np.maximum(
                    np.power(g_normalized[red_mask], inv_gamma) * 1.2,  # 先提升
                    target_white * 0.95  # 然后确保至少达到目标值的95%
                ),
                0, 1
            )
            b_adjusted[red_mask] = np.clip(
                np.maximum(
                    np.power(b_normalized[red_mask], inv_gamma) * 1.2,  # 先提升
                    target_white * 0.95  # 然后确保至少达到目标值的95%
                ),
                0, 1
            )
            
            # 让RGB三个通道都向目标白色值靠拢
            # 红色通道：如果太低，提升到至少0.85；如果太高，降低到0.92
            r_adjusted[red_mask] = np.clip(r_adjusted[red_mask], 0.85, target_white)
            
            # 让三个通道都向目标白色值靠拢（权重：75%目标值 + 25%当前值）
            # 这样能确保形成白色，而不是灰绿色
            r_adjusted[red_mask] = r_adjusted[red_mask] * 0.25 + target_white * 0.75
            g_adjusted[red_mask] = g_adjusted[red_mask] * 0.25 + target_white * 0.75
            b_adjusted[red_mask] = b_adjusted[red_mask] * 0.25 + target_white * 0.75
            
            # 最终裁剪确保在有效范围内
            r_adjusted[red_mask] = np.clip(r_adjusted[red_mask], 0.85, 1.0)  # 确保R至少0.85，避免偏绿
            g_adjusted[red_mask] = np.clip(g_adjusted[red_mask], 0.85, 1.0)  # 确保G至少0.85
            b_adjusted[red_mask] = np.clip(b_adjusted[red_mask], 0.85, 1.0)  # 确保B至少0.85
            
            # 最终裁剪确保在有效范围内
            r_adjusted[red_mask] = np.clip(r_adjusted[red_mask], 0, 1)
            g_adjusted[red_mask] = np.clip(g_adjusted[red_mask], 0, 1)
            b_adjusted[red_mask] = np.clip(b_adjusted[red_mask], 0, 1)
            
            # 调试信息：显示G、B通道调整
            if save_intermediate:
                print(f"绿色通道调整: 原始值范围 [{original_g_values.min():.3f}, {original_g_values.max():.3f}]")
                print(f"绿色通道调整: 调整后范围 [{g_adjusted[red_mask].min():.3f}, {g_adjusted[red_mask].max():.3f}]")
                print(f"蓝色通道调整: 原始值范围 [{original_b_values.min():.3f}, {original_b_values.max():.3f}]")
                print(f"蓝色通道调整: 调整后范围 [{b_adjusted[red_mask].min():.3f}, {b_adjusted[red_mask].max():.3f}]")
        
        # 将结果映射回0-255范围
        result_r = (r_adjusted * 255.0).astype(np.uint8)
        result_g = (g_adjusted * 255.0).astype(np.uint8)
        result_b = (b_adjusted * 255.0).astype(np.uint8)
        
        # 验证调整是否生效：比较原图和调整后的差异
        if save_intermediate and np.any(red_mask):
            original_r_uint8 = r.astype(np.uint8)
            diff_r = np.abs(result_r.astype(np.int16) - original_r_uint8.astype(np.int16))
            max_diff_r = np.max(diff_r[red_mask])
            mean_diff_r = np.mean(diff_r[red_mask])
            print(f"红色通道差异验证: 最大差异={max_diff_r}, 平均差异={mean_diff_r:.2f}")
            
            if max_diff_r == 0 and mean_diff_r == 0:
                print("警告: 红色通道调整后没有变化！请检查gamma值和red_mask检测")
        
        # 保存步骤3：调整后的各通道
        if save_intermediate:
            r_adjusted_path = os.path.join(intermediate_dir, f"{base_name}_step3_r_adjusted.jpg")
            g_adjusted_path = os.path.join(intermediate_dir, f"{base_name}_step3_g_adjusted.jpg")
            b_adjusted_path = os.path.join(intermediate_dir, f"{base_name}_step3_b_adjusted.jpg")
            cv2.imwrite(r_adjusted_path, result_r)
            cv2.imwrite(g_adjusted_path, result_g)
            cv2.imwrite(b_adjusted_path, result_b)
            print(f"步骤3 - 调整后的RGB通道已保存")
        
        # 合并通道：使用调整后的所有通道
        final_result = cv2.merge([result_b, result_g, result_r])
        
        # 保存最终结果
        if save_intermediate:
            merged_path = os.path.join(intermediate_dir, f"{base_name}_step3_merged.jpg")
            cv2.imwrite(merged_path, final_result)
            print(f"步骤3 - 合并后的结果已保存: {merged_path}")
        
        # 保存最终结果
        if output_path:
            cv2.imwrite(output_path, final_result)
            print(f"最终结果已保存到: {output_path}")
        
        return final_result
    
    def remove_seal_advanced(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        method: str = "red_channel"
    ) -> np.ndarray:
        """
        高级印章消除方法
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径
            method: 处理方法
                - "red_channel": 红色通道方法（默认）
                - "hsv": HSV颜色空间方法
                - "lab": LAB颜色空间方法
                
        Returns:
            处理后的图像
        """
        if method == "red_channel":
            return self.remove_seal(image_path, output_path)
        elif method == "hsv":
            return self._remove_seal_hsv(image_path, output_path)
        elif method == "lab":
            return self._remove_seal_lab(image_path, output_path)
        else:
            raise ValueError(f"未知的处理方法: {method}")
    
    def _remove_seal_hsv(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        使用HSV颜色空间消除印章
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径
            
        Returns:
            处理后的图像
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 第一步：使用BGR空间更准确地检测黑色文字区域（需要保护的区域）
        # 黑色文字特征：RGB三个通道值都很低且接近，或者整体很暗（可能包含红黑色叠加）
        b, g, r_bgr = cv2.split(img)
        
        # 方法1：标准黑色文字（RGB值都低且接近）
        black_text_mask_bgr1 = (r_bgr < 130) & (g < 130) & (b < 130) & \
                               (np.abs(r_bgr - g) < 40) & (np.abs(r_bgr - b) < 40) & \
                               (np.abs(g - b) < 40) & \
                               ((r_bgr + g + b) / 3.0 < 100)  # 平均亮度也要低
        
        # 方法2：整体很暗的区域（可能包含红黑色叠加，红色通道可能较高但整体很暗）
        # 对于红黑色叠加，红色通道可能较高，但绿色和蓝色通道低，整体亮度低
        # 限制红色通道值，避免检测到太红的区域
        black_text_mask_bgr2 = ((r_bgr + g + b) / 3.0 < 100) & \
                               (g < 100) & (b < 100) & \
                               (r_bgr < 100) & \
                               (np.abs(g - b) < 30)  # 绿色和蓝色接近，但红色可能较高（但不超过140）
        
        # 方法3：非常暗的区域（无论RGB分布如何，只要整体很暗）
        black_text_mask_bgr3 = ((r_bgr + g + b) / 3.0 < 80) & \
                               (np.maximum(np.maximum(r_bgr, g), b) < 120)
        
        # 合并BGR检测方法
        black_text_mask_bgr = black_text_mask_bgr1 | black_text_mask_bgr2 | black_text_mask_bgr3
        
        # 同时使用HSV空间检测黑色文字（作为补充，扩大范围）
        black_text_mask_hsv = (v < 130) & (s < 90)  # 扩大范围以包含更多区域
        
        # 合并两种检测方法
        black_text_mask = black_text_mask_bgr | black_text_mask_hsv
        
        # 对黑色文字区域进行形态学操作，扩大检测范围，确保包含红黑色叠加区域
        black_text_mask_uint8 = (black_text_mask.astype(np.uint8) * 255)
        kernel_black = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # 适度膨胀，确保边缘和红黑色叠加区域也被保护
        black_text_mask_uint8 = cv2.dilate(black_text_mask_uint8, kernel_black, iterations=2)
        black_text_mask = black_text_mask_uint8 > 0
        
        # 红色在HSV中的范围（扩大范围以检测边缘区域）
        # 红色在色环两端：扩大范围以包含更多红色变体
        red_mask1 = (h >= 0) & (h <= 20)  # 扩大范围从10到20
        red_mask2 = (h >= 160) & (h <= 180)
        red_mask = red_mask1 | red_mask2
        
        # 降低饱和度阈值，让边缘区域（低饱和度）也能被检测到
        # 边缘区域可能饱和度较低，所以降低阈值
        high_sat_mask = s > 30  # 从100降低到30，检测更多区域
        
        # 印章区域：红色且有一定饱和度（包括边缘）
        seal_mask = red_mask & high_sat_mask
        
        # 对检测到的区域进行形态学操作，扩大检测范围，确保边缘也被处理
        seal_mask_uint8 = (seal_mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # 膨胀操作，扩大检测区域，确保边缘也被包含
        seal_mask_uint8 = cv2.dilate(seal_mask_uint8, kernel, iterations=2)
        # 转换回布尔掩码
        seal_mask = seal_mask_uint8 > 0
        
        # 改进重叠区域检测：使用更精确的条件
        # 只检测直接重叠的区域（既是印章又是黑色文字），不扩大范围
        overlap_mask = seal_mask & black_text_mask
        
        # 区分纯印章区域和与黑色文字重叠的区域
        pure_seal_mask = seal_mask & (~overlap_mask)  # 纯印章区域（不是重叠区域）
        
        # 在印章区域，降低饱和度和提高明度，使其变白（红色更淡）
        s_result = s.copy()
        v_result = v.copy()
        
        # 对于纯印章区域：降低饱和度并提高明度，让颜色更接近白色
        s_result[pure_seal_mask] = np.clip(s_result[pure_seal_mask] * 0.02, 0, 255).astype(np.uint8)
        v_result[pure_seal_mask] = np.clip(v_result[pure_seal_mask] * 2.4, 0, 255).astype(np.uint8)
        
        # 对于重叠区域（红色和黑色文字重叠）：只去除红色，完全不改变明度
        # 重叠区域是红色和黑色的颜色叠加，只降低饱和度去除红色，明度保持不变
        s_result[overlap_mask] = np.clip(s_result[overlap_mask] * 0.02, 0, 255).astype(np.uint8)
        # 明度完全不改变，保持原始值
        # v_result[overlap_mask] 保持不变，不进行任何操作
        
        # 合并HSV通道
        hsv_result = cv2.merge([h, s_result, v_result])
        
        # 转换回BGR
        result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2BGR)
        
        # 在BGR空间中进一步处理重叠区域，确保只去除红色，保持黑色文字的明度
        # 对于重叠区域，只降低红色通道，保持绿色和蓝色通道不变
        result_b, result_g, result_r = cv2.split(result)
        # 对于重叠区域，将红色通道降低到接近绿色和蓝色通道的值（去除红色）
        # 但保持整体亮度不变
        overlap_avg = (result_g[overlap_mask].astype(np.float32) + result_b[overlap_mask].astype(np.float32)) / 2.0
        # 将红色通道设置为接近绿色和蓝色的平均值，去除红色但保持明度
        result_r[overlap_mask] = np.clip(overlap_avg, 0, 255).astype(np.uint8)
        # 重新合并BGR通道
        result = cv2.merge([result_b, result_g, result_r])
        
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"处理后的图像已保存到: {output_path}")
        
        return result
    
    # -*- coding: utf-8 -*-
    """
    印章消除模块 - 最佳版本
    使用HSV颜色空间和图像修复技术去除红色印章
    """


    def remove_seal_best(self, image_path, output_path=None):
        """
        使用HSV颜色空间和图像修复技术去除图像中的红色印章
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径（可选）
        
        Returns:
            处理后的图像（numpy数组）
        
        Raises:
            ValueError: 如果无法读取图像
        """
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError("无法读取图像")

        # HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 红色区域（最稳定）
        mask1 = (h < 10) & (s > 80) & (v < 230)
        mask2 = (h > 160) & (s > 80) & (v < 230)
        mask = ((mask1 | mask2).astype(np.uint8) * 255)

        # 形态学清洗
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 图像修复（真正去除印章）
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        if output_path:
            cv2.imwrite(output_path, result)

        return result



    def _remove_seal_lab(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        使用LAB颜色空间消除印章
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径
            
        Returns:
            处理后的图像
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 在LAB空间中，a通道表示红绿轴，红色区域a值较高
        # 印章通常是红色，a通道值较高
        red_mask = a > 130
        
        # 在印章区域，调整a通道值，使其接近中性（消除红色）
        a_result = a.copy()
        a_result[red_mask] = 128  # 中性值
        
        # 同时提高L通道（明度），使印章区域变白
        l_result = l.copy()
        l_result[red_mask] = np.clip(l_result[red_mask] * 1.3, 0, 255).astype(np.uint8)
        
        # 合并LAB通道
        lab_result = cv2.merge([l_result, a_result, b])
        
        # 转换回BGR
        result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
        
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"处理后的图像已保存到: {output_path}")
        
        return result


def main():
    """主函数示例"""
    remover = SealRemover()
    
    # 示例：处理图片
    input_image = "test.jpg"
    output_image = "output/ticket_no_seal.jpg"
    
    if not os.path.exists(input_image):
        print(f"错误: 找不到输入图片 {input_image}")
        return
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_image) if os.path.dirname(output_image) else ".", exist_ok=True)
    
    try:
        print(f"开始处理图片: {input_image}")
        result = remover.remove_seal(
            input_image,
            output_image,
            gamma=2.0,  # 色阶调整gamma值（>1），越大红色印章越接近白色，文字越接近黑色
            save_intermediate=True,  # 保存中间图片
            intermediate_dir=None  # 使用输出目录
        )
        print("处理完成！")
        
        # 也可以尝试其他方法
        result = remover.remove_seal_advanced(input_image, "output/ticket_no_seal_hsv.jpg", method="hsv")
        result = remover.remove_seal_advanced(input_image, "output/ticket_no_seal_lab.jpg", method="lab")
        # result = remover.remove_seal_best(input_image, "output/ticket_no_seal_best.jpg")
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

