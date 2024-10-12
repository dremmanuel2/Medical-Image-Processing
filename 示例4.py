import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']	# 显示中文
plt.rcParams['axes.unicode_minus'] = False		# 显示负号
def compute_area(image):
    return np.sum(image == 0)  # 计算黑色像素的数量


def calculate_hu_moments(image):
    image= np.uint8(np.where(image == 255, 0, 255))
    """
    计算给定二值图像的Hu矩。

    参数:
    image : numpy.ndarray
        输入的二值图像（黑白图像），非零值代表物体，零值代表背景。

    返回:
    hu_moments : numpy.ndarray
        归一化后的Hu矩数组。
    """
    # 计算图像的矩
    moments = cv2.moments(image)
    # 计算并返回归一化的Hu矩
    hu_moments = cv2.HuMoments(moments)
    # Hu矩通常需要进行归一化以提高比较的准确性
    hu_moments = np.array([10000 * hu[0] for hu in hu_moments])
    return hu_moments

def compute_perimeter_I(image):
    perimeter = 0
    height, width = image.shape

    for i in range(height):
        for j in range(width):
            if image[i, j] == 0:  # 当前像素为图案部分
                # 检查八个方向是否有背景
                neighbors = [
                    image[i - 1, j] if i > 0 else 255,  # 上
                    image[(i + 1) % height, j],  # 下
                    image[i, j - 1] if j > 0 else 255,  # 左
                    image[i, (j + 1) % width],  # 右
                    image[i - 1, j - 1] if i > 0 and j > 0 else 255,  # 左上
                    image[(i + 1) % height, (j + 1) % width],  # 右下
                    image[i - 1, (j + 1) % width] if i > 0 else 255,  # 右上
                    image[(i + 1) % height, j - 1] if j > 0 else 255  # 左下
                ]
                # 如果有相邻的背景，则认为是边界点

                if sum(n == 255 for n in neighbors)>=2:
                    perimeter += 1

    return perimeter


def compute_perimeter_II(image):
    perimeter = 0
    height, width = image.shape

    def get_neighbors(i, j):
        neighbors = [
            image[i - 1, j] if i > 0 else 255,  # 上方
            image[(i + 1) % height, j],  # 下方
            image[i, j - 1] if j > 0 else 255,  # 左侧
            image[i, (j + 1) % width],  # 右侧
            image[i - 1, j - 1] if i > 0 and j > 0 else 255,  # 左上方
            image[(i + 1) % height, (j + 1) % width],  # 右下方
            image[i - 1, (j + 1) % width] if i > 0 else 255,  # 右上方
            image[(i + 1) % height, j - 1] if j > 0 else 255  # 左下方
        ]
        return neighbors

    for i in range(height):
        for j in range(width):
            if image[i, j] == 0:  # 当前像素为图案部分
                neighbors = get_neighbors(i, j)
                # 如果有相邻的背景，则认为是边界点
                if sum(n == 255 for n in neighbors)>=2:
                    perimeter += 1  # 初始认为是直线边缘

                    # 检查是否为非直线边缘（角点）
                    diagonal_count = sum(n == 255 for n in neighbors[4:])
                    if diagonal_count > 0:
                        perimeter -= 1  # 减去之前添加的1
                        perimeter += np.sqrt(2)  # 添加根号2

    return perimeter


def find_largest_contour(image):
    """辅助函数用于找到图像中的最大轮廓"""
    image = np.uint8(np.where(image == 255, 0, 255))
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None, None, None
    c = max(contours, key=cv2.contourArea)
    return c


def calculate_features(image):
    """计算给定图像的主方向角度、偏心率及归一化的径向距离测量值"""
    # 寻找最大的轮廓
    largest_contour = find_largest_contour(image)

    if largest_contour is None:
        return None, None, None
    mu = cv2.moments(largest_contour)
    # 主方向角度
    angle = np.rad2deg(-0.5*np.arctan(2*mu['mu11']/(mu['mu20']-mu['mu02'])))
    # 偏心率
    eccentricity = ((mu['mu20']-mu['mu02'])**2+4*mu['mu11']**2)/(mu['mu20']+mu['mu02'])**2
    # 归一化的径向距离测量值
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    distances = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in largest_contour[:, 0]]
    normalized_distances = np.array(distances) / max(distances)
    m_1 = np.sum(normalized_distances)/len(normalized_distances)
    nui_2 = np.sum((normalized_distances-m_1)**2)/len(normalized_distances)
    nui_4 = np.sum((normalized_distances - m_1) ** 4 )/ len(normalized_distances)
    f_1 = (nui_2**0.5)/m_1
    f_2 = (nui_4 ** 0.25) / m_1
    return angle, eccentricity, f_2-f_1

# 创建一个 2x2 的图像网格
fig, axs = plt.subplots(1, 4, figsize=(10, 4))
for i, ax in enumerate(axs.flatten()):
    img0 = cv2.imread('homework/homework_4_{}.png'.format(i),0)
    img0_area = compute_area(img0)
    img0_perimeter_I = compute_perimeter_I(img0)
    img0_compactness_I= img0_perimeter_I**2/img0_area
    img0_compactness_norm_I = 1-4*np.pi/img0_compactness_I
    print('-----img{}方法一-----'.format(i))
    print('perimeter_I:{}'.format(img0_perimeter_I))
    print('area:{}'.format(img0_area))
    print('compactness_I:{:.2f}'.format(img0_compactness_I))
    print('compactness_norm_I:{:.3f}'.format(img0_compactness_norm_I))
    img0_perimeter_II = compute_perimeter_II(img0)
    img0_compactness_II= img0_perimeter_II**2/img0_area
    img0_compactness_norm_II = 1-4*np.pi/img0_compactness_II
    print('-----img{}方法二-----'.format(i))
    print('perimeter_II:{:.2f}'.format(img0_perimeter_II))
    print('area:{}'.format(img0_area))
    print('compactness_II:{:.2f}'.format(img0_compactness_II))
    print('compactness_norm_II:{:.3f}'.format(img0_compactness_norm_II))
    (img0_major_orientation ,img0_eccentricity,img0_radial_distance_measure)= calculate_features(img0)
    print('major_orientation:{:.4f}'.format(img0_major_orientation))
    print('eccentricity:{:.7f}'.format(img0_eccentricity))
    print('radial_distance_measure:{:.4f}'.format(img0_radial_distance_measure))
    # 显示图片
    ax.imshow(img0, cmap='gray')  # 假设是灰度图
    text = (
        f'-----img{i}方法一-----\n'
        f'周长: {img0_perimeter_I}\n'
        f'面积: {img0_area}\n'
        f'区域致密度: {img0_compactness_I:.2f}\n'
        f'归一化区域致密度: {img0_compactness_norm_I:.3f}\n\n'

        f'-----img{i}方法二-----\n'
        f'周长: {img0_perimeter_II:.2f}\n'
        f'面积: {img0_area}\n'
        f'区域致密度: {img0_compactness_II:.2f}\n'
        f'归一化区域致密度: {img0_compactness_norm_II:.3f}\n\n'

        f'形状主方向（°）: {img0_major_orientation:.4f}\n'
        f'偏心度: {img0_eccentricity:.7f}\n'
        f'径向距离测度f21: {img0_radial_distance_measure:.4f}'
    )
    ax.text(0.5, 0.15, text, transform=ax.transAxes,
            horizontalalignment='center', verticalalignment='top',
            fontsize=10)
    # 去掉坐标轴刻度
    ax.axis('off')
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.subplots_adjust(top=1.6, bottom=0, left=0.05, right=0.95, wspace=0.05)
plt.show()

img2 = cv2.imread('homework/homework_4_2.png',0)

# 平移变换
tx = 5  # 向右移动5像素
ty = 0  # 不向上或向下移动
M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
rows, cols = img2.shape[:2]
img2_move = cv2.warpAffine(img2, M_translate, (cols, rows), borderValue=255)
# 旋转变换
angle = 40  # 逆时针旋转40度
M_rotate = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)
img2_rotated = cv2.warpAffine(img2, M_rotate, (cols, rows), borderValue=255)
_, img2_rotated = cv2.threshold(img2_rotated, 127, 255, cv2.THRESH_BINARY)
# 缩放变换
scale = 2.0  # 放大两倍
img2_resized = cv2.resize(img2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
_, img2_resized = cv2.threshold(img2_resized, 127, 255, cv2.THRESH_BINARY)
img2_list = [img2,img2_move,img2_rotated,img2_resized]
img2_names = ['原图','向右平移5个像素','逆时针旋转40°','放大两倍']
fig, axs = plt.subplots(1, 4, figsize=(10, 3))
for i, ax in enumerate(axs.flatten()):
    img0 = img2_list[i]
    img0_area = compute_area(img0)
    img0_perimeter_I = compute_perimeter_I(img0)
    img0_compactness_I= img0_perimeter_I**2/img0_area
    img0_compactness_norm_I = 1-4*np.pi/img0_compactness_I
    print('perimeter_I:{}'.format(img0_perimeter_I))
    print('area:{}'.format(img0_area))
    print('compactness_I:{:.2f}'.format(img0_compactness_I))
    print('compactness_norm_I:{:.2f}'.format(img0_compactness_norm_I))
    # 显示图片
    ax.imshow(img0, cmap='gray')  # 假设是灰度图
    text = (
        f'-----{img2_names[i]}------\n'
        f'周长: {img0_perimeter_I}\n'
        f'面积: {img0_area}\n'
        f'区域致密度: {img0_compactness_I:.2f}\n'
        f'归一化区域致密度: {img0_compactness_norm_I:.3f}\n'
    )
    ax.text(0.5, 0.11, text, transform=ax.transAxes,
            horizontalalignment='center', verticalalignment='top',
            fontsize=10)
    # 去掉坐标轴刻度
    ax.axis('off')
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.subplots_adjust(top=1.5, bottom=0, left=0.05, right=0.95, wspace=0.05)
plt.show()

fig, axs = plt.subplots(1, 4, figsize=(10, 4))
for i, ax in enumerate(axs.flatten()):
    img0 = cv2.imread('homework/homework_4_{}.png'.format(i),0)
    img0_hu = calculate_hu_moments(img0)
    # 显示图片
    ax.imshow(img0, cmap='gray')  # 假设是灰度图
    text = [f'φ{x}:{img0_hu[x-1]}' for x in range(1, 8)]
    # 将列表中的所有元素连接成一个单一的字符串，元素之间用换行符分隔
    text_str = '\n'.join(text)

    ax.text(0.1, 0.15, text_str, transform=ax.transAxes,
            horizontalalignment='left', verticalalignment='top',
            fontsize=10)
    # 去掉坐标轴刻度
    ax.axis('off')
# plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.subplots_adjust(top=1.6, bottom=0, left=0.05, right=0.95, wspace=0.05)
plt.show()




