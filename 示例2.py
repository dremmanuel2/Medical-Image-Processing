import struct
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# 打开BMP文件
def delate_char(img,ib,ie,jb,je):
    img[ib:ie,jb:je] = img[25,313]
    return img



img = cv2.imread('homework/homework1_1.bmp',0)
# 显示图像
img = delate_char(img,2,33,218,289)
img = delate_char(img,191,311,20,37)
img = delate_char(img,215,244,55,452)
img = delate_char(img,201,220,475,490)
plt.imshow(img,cmap='gray')
plt.show()

img_width = img.shape[1]
img0 = img[:,5:img_width//3]
img1 = img[:,img_width//3-5:img_width*2//3-10]
img2 = img[:,img_width*2//3-5:img_width-1-10]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
img_ori_copy = img0.copy()
img_ori_copy = cv2.cvtColor(img_ori_copy, cv2.COLOR_GRAY2RGB)
img_ori_copy[:,:,:]=0
img_ori_copy0 = img_ori_copy.copy()
img_ori_copy1 = img_ori_copy.copy()
img_ori_copy2 = img_ori_copy.copy()
img_ori_copy0[:,:,0]=img0[:,:]
img_ori_copy1[:,:,1]=img1[:,:]
img_ori_copy2[:,:,2]=img2[:,:]
axs[0].imshow(img_ori_copy0)
axs[0].set_title('T1WI')
axs[1].imshow(img_ori_copy1)
axs[1].set_title('PDWI')
axs[2].imshow(img_ori_copy2)
axs[2].set_title('P2WI')
plt.tight_layout()
plt.show()

img_copy = img0.copy()
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
img_copy[:,:,0] = img_ori_copy0[:,:,0]
img_copy[:,:,1] = img_ori_copy1[:,:,1]
img_copy[:,:,2] = img_ori_copy2[:,:,2]
plt.imshow(img_copy)
plt.show()

# 定义颜色映射
colormap = {
    255: (255, 255, 255),  # 白色
    170: (255, 255, 0),  # 黄色
    85: (255, 0, 0),  # 红色
    0: (0, 0, 0),  # 黑色
}

# 创建一个空的RGB数组来存放颜色转换后的图像数据
colored_image = np.zeros((img0.shape[0], img0.shape[1], 3), dtype=np.uint8)

# 遍历每一个像素点，根据其灰度值映射到对应的颜色
for i in range(img0.shape[0]):
    for j in range(img0.shape[1]):
        gray_value = img0[i, j]
        if gray_value >= 240:
            color = colormap[255]
        elif gray_value >= 100:
            # 计算从白色到黄色之间的线性插值
            color = (
                255,
                int((255 - (gray_value - 170) * 255 / 85)),
                int((0 - (gray_value - 170) * 255 / 85))
            )
        elif gray_value > 0:
            # 计算从黄色到红色之间的线性插值
            color = (
                255,
                int((255 - (gray_value - 85) * 255 / 85)),
                0
            )
        else:
            color = colormap[0]

        colored_image[i, j] = color

# 创建一个空的RGB数组来存放颜色转换后的图像数据
rainbow_image = np.zeros((img0.shape[0], img0.shape[1], 3), dtype=np.uint8)

# 遍历每一个像素点，根据其灰度值映射到对应的颜色
for i in range(img0.shape[0]):
    for j in range(img0.shape[1]):
        gray_value = img0[i, j]
        # 归一化灰度值到0-255范围内
        normalized_value = int(gray_value / 255.0 * 255)

        # 根据灰度值确定颜色
        if normalized_value < 43:
            # 紫 -> 蓝
            color = (255 - int(normalized_value * 5.9), 0, 255)
        elif normalized_value < 130:
            # 蓝 -> 青
            color = (0, int((normalized_value - 43) * 5.9), 255)
        elif normalized_value < 173:
            # 青 -> 绿
            color = (0, 255, int(255 - (normalized_value - 130) * 5.9))
        elif normalized_value < 216:
            # 绿 -> 黄
            color = (int(255 - (normalized_value - 173) * 5.9), 255, 0)
        elif normalized_value < 259:
            # 黄 -> 红
            color = (255, int(255 - (normalized_value - 216) * 5.9), 0)
        else:
            # 红
            color = (255, 0, 0)

        rainbow_image[i, j] = color

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axs[0].imshow(colored_image)
axs[0].set_title('metal_image')
axs[1].imshow(rainbow_image)
axs[1].set_title('rainbow_image')
plt.tight_layout()
plt.show()
