import numpy as np
import cv2

class FastCausalLaneDetector:
    """高速时序因果车道线检测器"""
    
    def __init__(self, poly_degree=2, init_frames=8, threshold_diff=0.35, learning_rate=0.1):
        """
        参数:
            poly_degree: 多项式阶数
            init_frames: 初始化帧数
            threshold_diff: 预测差异阈值
            learning_rate: 多项式系数在线学习率
        """
        self.poly_degree = poly_degree
        self.init_frames = init_frames
        self.threshold_diff = threshold_diff
        self.lr = learning_rate
        
        # 多项式系数 (在线更新，不存储历史)
        # f(k) = a0 + a1*k + a2*k^2, g(b) = b0 + b1*b + b2*b^2
        self.left_f = np.zeros(poly_degree + 1)  # 左车道斜率变化多项式
        self.left_g = np.zeros(poly_degree + 1)  # 左车道截距变化多项式
        self.right_f = np.zeros(poly_degree + 1)
        self.right_g = np.zeros(poly_degree + 1)
        
        # 上一帧参数
        self.last_left_kb = None  # (k, b)
        self.last_right_kb = None
        
        # 初始化阶段临时存储
        self.init_left_data = []  # [(k, delta_k), ...]
        self.init_right_data = []
        
        # 状态
        self.is_initialized = False
        self.frame_count = 0
        
        # 预计算高斯核和Sobel核
        self._init_kernels()
        
    def _init_kernels(self):
        """预计算卷积核"""
        # 简化高斯核 5x5
        self.gauss_kernel = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=np.float32) / 256.0
        
        # Sobel核
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # 预计算霍夫变换的cos/sin表
        self.theta_res_fine = np.pi / 180
        self.theta_res_fast = np.pi / 60  # 快速模式3度分辨率
        
        self.thetas_fine = np.arange(np.pi/6, 5*np.pi/6, self.theta_res_fine)
        self.cos_fine = np.cos(self.thetas_fine)
        self.sin_fine = np.sin(self.thetas_fine)
        
        self.thetas_fast = np.arange(np.pi/6, 5*np.pi/6, self.theta_res_fast)
        self.cos_fast = np.cos(self.thetas_fast)
        self.sin_fast = np.sin(self.thetas_fast)
    
    def fast_conv2d(self, image, kernel):
        """快速卷积 - 使用分离或直接向量化"""
        k = kernel.shape[0] // 2
        padded = np.pad(image, k, mode='edge')
        
        # 使用stride_tricks加速
        shape = (image.shape[0], image.shape[1], kernel.shape[0], kernel.shape[1])
        strides = padded.strides * 2
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        
        return np.einsum('ijkl,kl->ij', windows, kernel)
    
    def gaussian_filter_fast(self, image):
        """快速高斯滤波"""
        return self.fast_conv2d(image.astype(np.float32), self.gauss_kernel).astype(np.uint8)
    
    def edge_detect_fast(self, image, fast_mode=False):
        """快速边缘检测"""
        img = image.astype(np.float32)
        
        # Sobel梯度
        gx = self.fast_conv2d(img, self.sobel_x)
        gy = self.fast_conv2d(img, self.sobel_y)
        
        mag = np.sqrt(gx*gx + gy*gy)
        
        # 简单阈值二值化
        threshold = 40 if fast_mode else 30
        edges = (mag > threshold).astype(np.uint8) * 255
        
        return edges
    
    def get_roi_points(self, edges, fast_mode=False):
        """提取ROI区域的边缘点"""
        h, w = edges.shape
        
        # ROI: 下半部分梯形区域
        y_start = int(h * 0.55)
        
        # 获取边缘点坐标
        ys, xs = np.where(edges[y_start:, :] > 0)
        ys = ys + y_start
        
        # ROI过滤: 梯形区域
        left_bound = (ys - h) * (-0.4 * w / (0.45 * h)) + 0.1 * w
        right_bound = (ys - h) * (0.4 * w / (0.45 * h)) + 0.9 * w
        
        mask = (xs > left_bound) & (xs < right_bound)
        xs, ys = xs[mask], ys[mask]
        
        # 快速模式下采样
        if fast_mode and len(xs) > 500:
            idx = np.random.choice(len(xs), 500, replace=False)
            xs, ys = xs[idx], ys[idx]
        
        return xs, ys
    
    def hough_detect(self, xs, ys, image_shape, fast_mode=False):
        """霍夫变换检测直线"""
        if len(xs) < 10:
            return None, None
        
        h, w = image_shape
        diag = int(np.sqrt(h*h + w*w))
        
        if fast_mode:
            thetas, cos_t, sin_t = self.thetas_fast, self.cos_fast, self.sin_fast
            rho_res = 4
            threshold = 15
        else:
            thetas, cos_t, sin_t = self.thetas_fine, self.cos_fine, self.sin_fine
            rho_res = 2
            threshold = 25
        
        num_rhos = 2 * diag // rho_res + 1
        accumulator = np.zeros((num_rhos, len(thetas)), dtype=np.int32)
        
        # 向量化投票
        for i, (ct, st) in enumerate(zip(cos_t, sin_t)):
            rhos = (xs * ct + ys * st).astype(np.int32)
            rho_idx = (rhos + diag) // rho_res
            valid = (rho_idx >= 0) & (rho_idx < num_rhos)
            np.add.at(accumulator[:, i], rho_idx[valid], 1)
        
        # 找峰值 - 分左右区域
        mid_theta = len(thetas) // 2
        
        # 左车道线 (theta < pi/2, 负斜率)
        left_acc = accumulator[:, :mid_theta]
        left_peaks = np.where(left_acc >= threshold)
        left_line = None
        if len(left_peaks[0]) > 0:
            max_idx = np.argmax(left_acc[left_peaks])
            rho_idx, theta_idx = left_peaks[0][max_idx], left_peaks[1][max_idx]
            rho = rho_idx * rho_res - diag
            theta = thetas[theta_idx]
            left_line = (rho, theta)
        
        # 右车道线 (theta > pi/2, 正斜率)
        right_acc = accumulator[:, mid_theta:]
        right_peaks = np.where(right_acc >= threshold)
        right_line = None
        if len(right_peaks[0]) > 0:
            max_idx = np.argmax(right_acc[right_peaks])
            rho_idx, theta_idx = right_peaks[0][max_idx], right_peaks[1][max_idx]
            rho = rho_idx * rho_res - diag
            theta = thetas[mid_theta + theta_idx]
            right_line = (rho, theta)
        
        return left_line, right_line
    
    def rho_theta_to_kb(self, rho, theta):
        """(ρ,θ) -> (k,b)"""
        if abs(np.sin(theta)) < 0.01:
            return None, None
        k = -np.cos(theta) / np.sin(theta)
        b = rho / np.sin(theta)
        return k, b
    
    def poly_eval(self, coeffs, x):
        """计算多项式值 f(x) = c0 + c1*x + c2*x^2 + ..."""
        result = 0.0
        for i, c in enumerate(coeffs):
            result += c * (x ** i)
        return result
    
    def poly_gradient(self, x, degree):
        """多项式对系数的梯度 [1, x, x^2, ...]"""
        return np.array([x ** i for i in range(degree + 1)])
    
    def update_poly_online(self, coeffs, x, target_delta, lr):
        """
        在线更新多项式系数
        误差 = target_delta - f(x)
        梯度下降: coeffs += lr * error * grad
        """
        pred = self.poly_eval(coeffs, x)
        error = target_delta - pred
        grad = self.poly_gradient(x, len(coeffs) - 1)
        
        # 梯度更新，带衰减
        coeffs += lr * error * grad * 0.1
        
        # 限制系数范围防止发散
        coeffs = np.clip(coeffs, -1.0, 1.0)
        return coeffs
    
    def fit_initial_poly(self, data):
        """初始化阶段拟合多项式"""
        if len(data) < self.poly_degree + 1:
            return np.zeros(self.poly_degree + 1)
        
        x_data = np.array([d[0] for d in data])
        y_data = np.array([d[1] for d in data])
        
        try:
            # 最小二乘拟合
            A = np.vstack([x_data ** i for i in range(self.poly_degree + 1)]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, y_data, rcond=None)
            return np.clip(coeffs, -1.0, 1.0)
        except:
            return np.zeros(self.poly_degree + 1)
    
    def predict_kb(self, last_kb, f_coeffs, g_coeffs):
        """用多项式预测下一帧参数"""
        if last_kb is None or last_kb[0] is None:
            return None
        
        k, b = last_kb
        delta_k = self.poly_eval(f_coeffs, k)
        delta_b = self.poly_eval(g_coeffs, b)
        
        return (k + delta_k, b + delta_b)
    
    def check_and_fuse(self, pred_kb, det_kb):
        """检查预测质量并融合"""
        if pred_kb is None or det_kb is None or det_kb[0] is None:
            return det_kb, False
        
        if pred_kb[0] is None:
            return det_kb, True
        
        pk, pb = pred_kb
        dk, db = det_kb
        
        # 相对误差
        k_err = abs(pk - dk) / (abs(dk) + 0.1)
        b_err = abs(pb - db) / (abs(db) + 1.0)
        
        if k_err < self.threshold_diff and b_err < self.threshold_diff:
            # 融合: 加权平均，预测权重随误差减小
            w = 0.5 - 0.3 * (k_err + b_err)
            w = max(0.2, min(0.5, w))
            fused_k = w * pk + (1 - w) * dk
            fused_b = w * pb + (1 - w) * db
            return (fused_k, fused_b), True
        else:
            return det_kb, False
    
    def reset(self):
        """重置状态"""
        self.left_f = np.zeros(self.poly_degree + 1)
        self.left_g = np.zeros(self.poly_degree + 1)
        self.right_f = np.zeros(self.poly_degree + 1)
        self.right_g = np.zeros(self.poly_degree + 1)
        self.last_left_kb = None
        self.last_right_kb = None
        self.init_left_data = []
        self.init_right_data = []
        self.is_initialized = False
        self.frame_count = 0
    
    def process_frame(self, frame):
        """处理单帧"""
        self.frame_count += 1
        fast_mode = self.is_initialized
        
        # 1. 灰度
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # 2. 高斯滤波
        if fast_mode:
            # 快速模式: 下采样处理
            small = cv2.resize(gray, None, fx=0.5, fy=0.5)
            filtered = self.gaussian_filter_fast(small)
            edges = self.edge_detect_fast(filtered, fast_mode=True)
            # 上采样边缘图
            edges = cv2.resize(edges, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            filtered = self.gaussian_filter_fast(gray)
            edges = self.edge_detect_fast(filtered, fast_mode=False)
        
        # 3. 提取ROI边缘点
        xs, ys = self.get_roi_points(edges, fast_mode)
        
        # 4. 霍夫变换
        left_line, right_line = self.hough_detect(xs, ys, gray.shape, fast_mode)
        
        # 转换为(k,b)
        left_kb = self.rho_theta_to_kb(*left_line) if left_line else (None, None)
        right_kb = self.rho_theta_to_kb(*right_line) if right_line else (None, None)
        
        final_left_kb = left_kb
        final_right_kb = right_kb
        
        # 5. 因果预测与融合
        if self.is_initialized:
            # 预测
            pred_left = self.predict_kb(self.last_left_kb, self.left_f, self.left_g)
            pred_right = self.predict_kb(self.last_right_kb, self.right_f, self.right_g)
            
            # 融合检查
            final_left_kb, left_ok = self.check_and_fuse(pred_left, left_kb)
            final_right_kb, right_ok = self.check_and_fuse(pred_right, right_kb)
            
            # 如果预测失败，重置
            if not left_ok and not right_ok and self.frame_count > self.init_frames + 5:
                print(f"Frame {self.frame_count}: Prediction failed, resetting...")
                self.reset()
                return self.process_frame(frame)
            
            # 在线更新多项式
            if left_kb[0] is not None and self.last_left_kb is not None and self.last_left_kb[0] is not None:
                delta_k = left_kb[0] - self.last_left_kb[0]
                delta_b = left_kb[1] - self.last_left_kb[1]
                self.left_f = self.update_poly_online(self.left_f, self.last_left_kb[0], delta_k, self.lr)
                self.left_g = self.update_poly_online(self.left_g, self.last_left_kb[1], delta_b, self.lr)
            
            if right_kb[0] is not None and self.last_right_kb is not None and self.last_right_kb[0] is not None:
                delta_k = right_kb[0] - self.last_right_kb[0]
                delta_b = right_kb[1] - self.last_right_kb[1]
                self.right_f = self.update_poly_online(self.right_f, self.last_right_kb[0], delta_k, self.lr)
                self.right_g = self.update_poly_online(self.right_g, self.last_right_kb[1], delta_b, self.lr)
        
        else:
            # 初始化阶段: 收集数据
            if left_kb[0] is not None and self.last_left_kb is not None and self.last_left_kb[0] is not None:
                delta_k = left_kb[0] - self.last_left_kb[0]
                delta_b = left_kb[1] - self.last_left_kb[1]
                self.init_left_data.append((self.last_left_kb[0], delta_k))
                self.init_left_data.append((self.last_left_kb[1], delta_b))
            
            if right_kb[0] is not None and self.last_right_kb is not None and self.last_right_kb[0] is not None:
                delta_k = right_kb[0] - self.last_right_kb[0]
                delta_b = right_kb[1] - self.last_right_kb[1]
                self.init_right_data.append((self.last_right_kb[0], delta_k))
                self.init_right_data.append((self.last_right_kb[1], delta_b))
            
            # 检查是否可以初始化
            if self.frame_count >= self.init_frames:
                if len(self.init_left_data) >= self.poly_degree + 1:
                    # 分离k和b的数据
                    k_data = self.init_left_data[::2]
                    b_data = self.init_left_data[1::2]
                    self.left_f = self.fit_initial_poly(k_data)
                    self.left_g = self.fit_initial_poly(b_data)
                
                if len(self.init_right_data) >= self.poly_degree + 1:
                    k_data = self.init_right_data[::2]
                    b_data = self.init_right_data[1::2]
                    self.right_f = self.fit_initial_poly(k_data)
                    self.right_g = self.fit_initial_poly(b_data)
                
                self.is_initialized = True
                # 清空初始化数据
                self.init_left_data = []
                self.init_right_data = []
                print(f"Frame {self.frame_count}: Initialized, switching to fast mode")
        
        # 更新上一帧参数
        if final_left_kb[0] is not None:
            self.last_left_kb = final_left_kb
        if final_right_kb[0] is not None:
            self.last_right_kb = final_right_kb
        
        # 6. 绘制结果
        result = self.draw_lanes(frame, final_left_kb, final_right_kb)
        
        return result
    
    def draw_lanes(self, frame, left_kb, right_kb):
        """绘制车道线"""
        result = frame.copy()
        h, w = frame.shape[:2]
        
        y1, y2 = h, int(h * 0.6)
        
        if left_kb[0] is not None:
            k, b = left_kb
            if abs(k) > 0.01:
                x1 = int((y1 - b) / k)
                x2 = int((y2 - b) / k)
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        if right_kb[0] is not None:
            k, b = right_kb
            if abs(k) > 0.01:
                x1 = int((y1 - b) / k)
                x2 = int((y2 - b) / k)
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 状态显示
        mode = "FAST" if self.is_initialized else "INIT"
        cv2.putText(result, f"{mode} F:{self.frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return result

def process_video(input_path, output_path):
    """处理视频"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open {input_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    detector = FastCausalLaneDetector(
        poly_degree=2,
        init_frames=8,
        threshold_diff=0.35,
        learning_rate=0.15
    )
    
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.process_frame(frame)
        out.write(result)
        
        idx += 1
        if idx % 50 == 0:
            print(f"Processed {idx}/{total}")
    
    cap.release()
    out.release()
    print(f"Saved to {output_path}")
    
if __name__ == "__main__":
    process_video("input_video.mp4", "outputFast.mp4")