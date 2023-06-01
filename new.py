import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 计算每个像素的最小成本
def findCostArr(edgeImg):
    rows, cols = edgeImg.shape
    cost = np.zeros((rows, cols))
    cost[-1, :] = edgeImg[-1, :]
    for i in range(rows - 2, -1, -1):
        for j in range(cols):
            if j == 0:
                cost[i, j] = edgeImg[i, j] + min(cost[i + 1, j], cost[i + 1, j + 1])
            elif j == cols - 1:
                cost[i, j] = edgeImg[i, j] + min(cost[i + 1, j], cost[i + 1, j - 1])
            else:
                cost[i, j] = edgeImg[i, j] + min(cost[i + 1, j - 1], cost[i + 1, j], cost[i + 1, j + 1])
    return cost

# 找到要删除的Seam
def findVerticalSeam(edgeImg):
    rows= edgeImg.shape[0]
    cols = edgeImg.shape[1]
    cost = findCostArr(edgeImg)
    seam = np.zeros(rows, dtype=np.int32)
    j = np.argmin(cost[0, :])
    seam[0] = j
    for i in range(1, rows):
        if j == 0:
            j = np.argmin(cost[i, j:j + 2])
        elif j == cols - 1:
            j = np.argmin(cost[i, j - 1:j + 1]) + j - 1
        else:
            j = np.argmin(cost[i, j - 1:j + 2]) + j - 1
        seam[i] = j
    return seam

# 找到要删除的Seam
def findHorizontalSeam(edgeImg):
    edgeImg = np.transpose(edgeImg, (1, 0))
    seam = findVerticalSeam(edgeImg)
    return seam

# 删除水平方向上的Seam
def removeHorizontalSeam(img, seam):
    r,c,_ = img.shape
    newImg = np.zeros((r,c,3))
    for i,j in enumerate(seam):
        newImg[i,0:j,:] = img[i,0:j,:]
        newImg[i,j:c-1,:] = img[i,j+1:c,:]
    return newImg[:,:-1,:].astype(np.int32)

# 删除垂直方向上的Seam
def removeVerticalSeam(img, seam):
    rows, cols, _ = img.shape
    newImg = np.zeros((rows, cols - 1, 3), dtype=np.int32)
    for i in range(rows):
        newImg[i, :, :] = np.delete(img[i, :, :], seam[i], axis=0)
    return newImg

# 增加水平方向上的Seam
def addHorizontalSeam(img, edgeImg, seam):
    rows, cols, _ = img.shape
    newImg = np.zeros((rows + 1, cols, 3), dtype=np.int32)
    newEdgeImg = np.zeros((rows + 1, cols), dtype=np.int32)
    for j in range(cols):
        row = seam[j]
        newImg[:row + 1, j, :] = img[:row + 1, j, :]
        newImg[row + 1:, j, :] = img[row:, j, :]
        newEdgeImg[:row + 1, j] = edgeImg[:row + 1, j]
        newEdgeImg[row + 1, j] = edgeImg[row, j]
        newEdgeImg[row + 2:, j] = edgeImg[row + 1:, j]
    return newImg, newEdgeImg

# 增加垂直方向上的Seam
def addVerticalSeam(img, edgeImg, seam):
    img = np.transpose(img, (1, 0, 2))
    edgeImg = np.transpose(edgeImg, (1, 0))
    newImg, newEdgeImg = addHorizontalSeam(img, edgeImg, seam)
    newImg = np.transpose(newImg, (1, 0, 2))
    newEdgeImg = np.transpose(newEdgeImg, (1, 0))
    return newImg, newEdgeImg

# 计算图像的边缘
def getEdge(img):
    sX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    edgeX = fastConvolve(img, sX)
    edgeY = fastConvolve(img, sY)
    edgeImg = np.sqrt(np.square(edgeX) + np.square(edgeY))
    return edgeImg

# 快速卷积
def fastConvolve(img, kernel):
    fImg = np.fft.fft2(img, axes=(0, 1))
    fKernel = np.fft.fft2(kernel, s=img.shape[:2], axes=(0, 1))
    return np.real(np.fft.ifft2(fImg * fKernel, axes=(0, 1)))

def rgbToGrey(arr):
    greyVal = np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140])
    return np.round(greyVal).astype(np.int32)

# 读取图像
img = Image.open("input.png")
img = np.array(img)

# 输入要删除或增加的方向
dir = input("Enter direction (h/v): ")
# 输入要删除或增加的像素数
n = int(input("Enter number of pixels to be removed: "))

# 将图像转换为灰度图像
greyImg = rgbToGrey(img)

# 计算边缘图像
edgeImg = getEdge(greyImg)

# 删除或增加像素
if dir == "h":
    for i in range(n):
        seam = findHorizontalSeam(edgeImg)
        img = removeHorizontalSeam(greyImg, seam)
        edgeImg = removeHorizontalSeam(edgeImg, seam)
elif dir == "v":
    for i in range(n):
        seam = findVerticalSeam(edgeImg)
        img = removeVerticalSeam(img, seam)
        edgeImg = removeVerticalSeam(edgeImg, seam)
else:
    print("Invalid direction. Please enter 'h' or 'v'.")

# 显示图像
plt.imshow(img)
plt.show()

# 保存图像
Image.fromarray(img.astype(np.uint8)).save("output.png")
