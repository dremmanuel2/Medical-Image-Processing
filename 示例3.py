import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']	# 显示中文
plt.rcParams['axes.unicode_minus'] = False		# 显示负号

img1043 = cv2.imread('homework/homework_3_0.png',0)
img1077 = cv2.imread('homework/homework_3_1.png',0)
img1060 = cv2.imread('homework/homework_3_2.png',0)

fig,axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
axs[0].imshow(img1043, cmap = 'gray')
axs[0].set_title('T1043')
axs[1].imshow(img1077, cmap = 'gray')
axs[1].set_title('T1077')
axs[2].imshow(img1060, cmap = 'gray')
axs[2].set_title('T1060')
plt.tight_layout()
plt.show()

img1043_sub = np.zeros((74,74))
img1043_sub[:,:]=img1043[::3,::3]
img1077_sub = np.zeros((74,74))
img1077_sub[:,:]=img1077[::3,::3]
fig,axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
axs[0,0].imshow(img1043, cmap = 'gray')
axs[0,0].set_title('T1043原始图像')
axs[0,1].imshow(img1043_sub, cmap = 'gray')
axs[0,1].set_title('T1043采样后图像')
axs[1,0].imshow(img1077, cmap = 'gray')
axs[1,0].set_title('T1077原始图像')
axs[1,1].imshow(img1077_sub, cmap = 'gray')
axs[1,1].set_title('T1077采样后图像')
plt.tight_layout()
plt.show()

img1043_sub_NEAREST = cv2.resize(img1043_sub, dsize=None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)#最邻近插值
img1043_sub_LINEAR = cv2.resize(img1043_sub, dsize=None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)#双线性插值
img1043_sub_CUBIC = cv2.resize(img1043_sub, dsize=None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)#三次多项式插值
img1043_sub_NEAREST_dif = 255 - np.absolute(img1043 - img1043_sub_NEAREST)
img1043_sub_LINEAR_dif = 255 - np.absolute(img1043 - img1043_sub_LINEAR)
img1043_sub_CUBIC_dif = 255 - np.absolute(img1043 - img1043_sub_CUBIC)
fig,axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
axs[0,0].imshow(img1043, cmap = 'gray')
axs[0,0].set_title('T1043原始图像')
axs[0,1].imshow(img1043_sub_NEAREST, cmap = 'gray')
axs[0,1].set_title('最邻近插值')
axs[0,2].imshow(img1043_sub_LINEAR, cmap = 'gray')
axs[0,2].set_title('双线性插值')
axs[0,3].imshow(img1043_sub_CUBIC, cmap = 'gray')
axs[0,3].set_title('三次多项式插值')
axs[1,1].imshow(img1043_sub_NEAREST_dif, cmap = 'gray')
axs[1,1].set_title('最邻近插值与原图差值')
axs[1,2].imshow(img1043_sub_LINEAR_dif, cmap = 'gray')
axs[1,2].set_title('双线性插值与原图差值')
axs[1,3].imshow(img1043_sub_CUBIC_dif, cmap = 'gray')
axs[1,3].set_title('三次多项式插值与原图差值')
plt.tight_layout()
plt.show()

img1077_sub_NEAREST = cv2.resize(img1077_sub, dsize=None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)#最邻近插值
img1077_sub_LINEAR = cv2.resize(img1077_sub, dsize=None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)#双线性插值
img1077_sub_CUBIC = cv2.resize(img1077_sub, dsize=None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)#3次插值
img1077_sub_NEAREST_dif = 255 - np.absolute(img1077 - img1077_sub_NEAREST)
img1077_sub_LINEAR_dif = 255 - np.absolute(img1077 - img1077_sub_LINEAR)
img1077_sub_CUBIC_dif = 255 - np.absolute(img1077 - img1077_sub_CUBIC)
fig,axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
axs[0,0].imshow(img1043, cmap = 'gray')
axs[0,0].set_title('T1043原始图像')
axs[0,1].imshow(img1077_sub_NEAREST, cmap = 'gray')
axs[0,1].set_title('最邻近插值')
axs[0,2].imshow(img1077_sub_LINEAR, cmap = 'gray')
axs[0,2].set_title('双线性插值')
axs[0,3].imshow(img1077_sub_CUBIC, cmap = 'gray')
axs[0,3].set_title('三次多项式插值')
axs[1,1].imshow(img1077_sub_NEAREST_dif, cmap = 'gray')
axs[1,1].set_title('最邻近插值与原图差值')
axs[1,2].imshow(img1077_sub_LINEAR_dif, cmap = 'gray')
axs[1,2].set_title('双线性插值与原图差值')
axs[1,3].imshow(img1077_sub_CUBIC_dif, cmap = 'gray')
axs[1,3].set_title('三次多项式插值与原图差值')
plt.tight_layout()
plt.show()

img1060_pred = cv2.addWeighted(img1043_sub_LINEAR, 0.5, img1077_sub_LINEAR, 0.5, 0)
img1060_pred_dif = 255 - np.absolute(img1060 - img1060_pred)
fig,axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
axs[0].imshow(img1060, cmap = 'gray')
axs[0].set_title('T1060原始图像')
axs[1].imshow(img1060_pred, cmap = 'gray')
axs[1].set_title('T1060线性插值图像')
axs[2].imshow(img1060_pred_dif, cmap = 'gray')
axs[2].set_title('差值图像')
plt.tight_layout()
plt.show()
