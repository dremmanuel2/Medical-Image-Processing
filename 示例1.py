import struct
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 打开BMP文件
def delate_char(img,ib,ie,jb,je):
    img[ib:ie,jb:je] = img[25,313]
    return img
with open('homework/homework1_1.bmp', 'rb') as file:
    # 读取文件头 (14字节)
    file_header = file.read(14)
    (type_id, file_size, reserved1, reserved2, offset) = struct.unpack('<2sI2HI', file_header)

    # 读取信息头 (至少40字节)
    info_header = file.read(40)
    (biSize, biWidth, biHeight, biPlanes, bitBitCount, biCompression, biSizeImage,
     biXPelsPerMeter, biYPelsPerMeter, biClrUsed, biClrImportant) = struct.unpack(
        '<IIIhhiIIIII', info_header)

    print("Type ID:", type_id.decode('utf-8'))
    print("File Size:", file_size)
    print("reserved1:", reserved1)
    print("reserved2:", reserved2)
    print("Offset:", offset)
    print("biSize:", biSize)
    print("biWidth:", biWidth)
    print("biHeight:", biHeight)
    print("biPlanes:", biPlanes)
    print("bitBitCount:", bitBitCount)
    print("biCompression:", biCompression)
    print("biSizeImage:", biSizeImage)
    print("biXPelsPerMeter:", biXPelsPerMeter)
    print("biYPelsPerMeter:", biYPelsPerMeter)
    print("biClrUsed:", biClrUsed)
    print("biClrImportant:", biClrImportant)


# 打开BMP文件并转换为图像对象
img = cv2.imread('homework/homework1_1.bmp',0)
# 显示图像
img = delate_char(img,2,33,218,289)
img = delate_char(img,191,311,20,37)
img = delate_char(img,215,244,55,452)
img = delate_char(img,201,220,475,490)
plt.imshow(img,cmap='gray')
plt.show()

img_width = img.shape[1]
img1 = img[:,:img_width//3]
img2 = img[:,img_width//3:img_width*2//3]
img3 = img[:,img_width*2//3:img_width]
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
axs[0].imshow(img1,cmap='gray')
axs[0].set_title('T1WI')
axs[1].imshow(img2,cmap='gray')
axs[1].set_title('PDWI')
axs[2].imshow(img3,cmap='gray')
axs[2].set_title('P2WI')
plt.tight_layout()
plt.show()

histogram1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axs[0].plot(histogram1, color='r')
axs[0].set_title('T1')
histogram1[:20]=0
axs[1].plot(histogram1, color='r')
axs[1].set_title('T1_without_BG')
plt.tight_layout()
plt.show()

histogram2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axs[0].plot(histogram2, color='r')
axs[0].set_title('PD')
histogram2[:20]=0
axs[1].plot(histogram2, color='r')
axs[1].set_title('PD_without_BG')
plt.tight_layout()
plt.show()

histogram3 = cv2.calcHist([img3], [0], None, [256], [0, 256])
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axs[0].plot(histogram3, color='r')
axs[0].set_title('T2')
histogram3[:20]=0
axs[1].plot(histogram3, color='r')
axs[1].set_title('T2_without_BG')
plt.tight_layout()
plt.show()