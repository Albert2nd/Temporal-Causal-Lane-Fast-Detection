import numpy as np
import cv2
# ==================== 1. 灰度转换（可调库）====================
def to_grayscale(image):
    """将彩色图像转换为灰度图像"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ==================== 2. 滤波（不能调库）====================
def gaussian_kernel(size, sigma):
    """生成高斯核"""
    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    sum_val = 0
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            sum_val += kernel[i, j]
    kernel /= sum_val  # 归一化
    return kernel
def apply_filter(image, kernel):
    """手动实现卷积操作"""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    # 边界填充
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return np.clip(output, 0, 255).astype(np.uint8)
def gaussian_blur(image, kernel_size=5, sigma=1.4):
    """高斯滤波（不调库）"""
    kernel = gaussian_kernel(kernel_size, sigma)
    return apply_filter(image, kernel)
# ==================== 3. 边缘检测（不能调库）====================
def sobel_edge_detection(image):
    """Sobel边缘检测"""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    grad_x = apply_filter(image, sobel_x)
    grad_y = apply_filter(image, sobel_y)
    magnitude = np.sqrt(grad_x.astype(np.float64)**2 + grad_y.astype(np.float64)**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude, grad_x, grad_y
def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """Canny边缘检测（不调库）"""
    # 1. 高斯滤波
    blurred = gaussian_blur(image, 5, 1.4)
    # 2. 计算梯度
    magnitude, grad_x, grad_y = sobel_edge_detection(blurred)
    # 3. 计算梯度方向
    angle = np.arctan2(grad_y.astype(np.float64), grad_x.astype(np.float64)) * 180 / np.pi
    angle[angle < 0] += 180
    # 4. 非极大值抑制
    h, w = magnitude.shape
    nms = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            q, r = 255, 255
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif 22.5 <= a < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif 67.5 <= a < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif 112.5 <= a < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                nms[i, j] = magnitude[i, j]
    # 5. 双阈值检测和边缘连接
    strong = 255
    weak = 75
    result = np.zeros((h, w), dtype=np.uint8)
    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms >= low_threshold) & (nms < high_threshold))
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    # 边缘连接
    for i in range(1, h-1):
        for j in range(1, w-1):
            if result[i, j] == weak:
                if np.any(result[i-1:i+2, j-1:j+2] == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    return result
# ==================== 4. ROI提取（可调库）====================
def region_of_interest(image):
    """提取感兴趣区域"""
    h, w = image.shape[:2]
    # 定义梯形ROI区域
    vertices = np.array([
        [(int(w*0.1), h),
         (int(w*0.45), int(h*0.6)),
         (int(w*0.55), int(h*0.6)),
         (int(w*0.9), h)]
    ], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
# ==================== 5. 霍夫变换（不能调库）====================
def hough_transform(edge_image, rho_resolution=1, theta_resolution=np.pi/180, threshold=50):
    """霍夫变换检测直线（不调库）"""
    h, w = edge_image.shape
    diag_len = int(np.sqrt(h**2 + w**2))
    # 参数空间
    rhos = np.arange(-diag_len, diag_len, rho_resolution)
    thetas = np.arange(0, np.pi, theta_resolution)
    # 累加器
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    # 找到边缘点
    y_idxs, x_idxs = np.nonzero(edge_image)
    # 投票
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(len(thetas)):
            rho = int(x * cos_thetas[t_idx] + y * sin_thetas[t_idx]) + diag_len
            accumulator[rho, t_idx] += 1
    # 提取峰值
    lines = []
    for rho_idx in range(accumulator.shape[0]):
        for theta_idx in range(accumulator.shape[1]):
            if accumulator[rho_idx, theta_idx] >= threshold:
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]
                lines.append((rho, theta))
    return lines
def draw_lines(image, lines, color=(0, 255, 255), thickness=3):
    """绘制检测到的直线"""
    h, w = image.shape[:2]
    left_lines = []
    right_lines = []
    for rho, theta in lines:
        # 过滤水平线
        if abs(theta - np.pi/2) < np.pi/6:
            continue
        # 根据斜率分类左右车道线
        if theta < np.pi/2:
            right_lines.append((rho, theta))
        else:
            left_lines.append((rho, theta))
    # 绘制左车道线
    if left_lines:
        rho, theta = np.mean(left_lines, axis=0)
        draw_line(image, rho, theta, color, thickness, h)
    # 绘制右车道线
    if right_lines:
        rho, theta = np.mean(right_lines, axis=0)
        draw_line(image, rho, theta, color, thickness, h)
    return image
def draw_line(image, rho, theta, color, thickness, h):
    """绘制单条直线"""
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # 计算与图像边界的交点
    y1 = h
    y2 = int(h * 0.6)
    if abs(b) > 1e-6:
        x1 = int((rho - y1 * b) / a)
        x2 = int((rho - y2 * b) / a)
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
# ==================== 主函数：车道线检测 ====================
def detect_lane_lines(frame):
    """对单帧图像进行车道线检测"""
    # 1. 灰度转换
    gray = to_grayscale(frame)
    # 2. 高斯滤波（不调库）
    blurred = gaussian_blur(gray, kernel_size=5, sigma=1.4)
    # 3. 边缘检测（不调库）
    edges = canny_edge_detection(blurred, low_threshold=50, high_threshold=150)
    # 4. 提取ROI（可调库）
    roi = region_of_interest(edges)
    # 5. 霍夫变换检测直线（不调库）
    lines = hough_transform(roi, threshold=80)
    # 6. 绘制车道线
    result = frame.copy()
    if lines:
        draw_lines(result, lines, color=(0, 255, 255), thickness=5)
    return result
# ==================== 6. 视频处理（可调库）====================
def process_video(input_path, output_path):
    """处理视频文件"""
    cap = cv2.VideoCapture(input_path)
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 车道线检测
        result = detect_lane_lines(frame)
        # 写入输出视频
        out.write(result)
        frame_count += 1
        print(f"\r处理进度: {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)", end="")
    cap.release()
    out.release()
    print(f"\n视频处理完成，保存至: {output_path}")
# ==================== 运行主程序 ====================
if __name__ == "__main__":
    input_video = "input_video.mp4"  # 输入视频路径
    output_video = "output_video.mp4"  # 输出视频路径
    process_video(input_video, output_video)
