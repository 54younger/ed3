from PIL import Image
import numpy as np


def fastConvolve(img, ker):
    imgF = np.fft.rfft2(img)
    kerF = np.fft.rfft2(ker, img.shape)
    return np.fft.irfft2(imgF * kerF)


def getEdge(greyImg):
    sX = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sY = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    edgeH = fastConvolve(greyImg, sX)
    edgeV = fastConvolve(greyImg, sY)

    return np.sqrt(np.square(edgeH) + np.square(edgeV))


def findCostArr(edgeImg):
    r, c = edgeImg.shape
    cost = np.zeros(edgeImg.shape)
    cost[r - 1, :] = edgeImg[r - 1, :]
    print(type(cost), type(edgeImg))
    for i in range(r - 2, -1, -1):
        for j in range(c):
            c1, c2 = max(j - 1, 0), min(c, j + 2)
            cost[i][j] = edgeImg[i][j] + cost[i + 1, c1:c2].min()

    return cost


def findSeam(cost):
    r, c = cost.shape

    path = []
    j = cost[0].argmin()  # 返回最小值的索引
    path.append(j)

    for i in range(r - 1):
        c1, c2 = max(j - 1, 0), min(c, j + 2)
        j = max(j - 1, 0) + cost[i + 1, c1:c2].argmin()
        path.append(j)

    return path


def removeSeam(img, path):
    print(img.shape)
    r, c, _ = img.shape
    newImg = np.zeros((r, c, 3))
    for i, j in enumerate(path):
        newImg[i, 0:j, :] = img[i, 0:j, :]
        newImg[i, j : c - 1, :] = img[i, j + 1 : c, :]
    return newImg[:, :-1, :].astype(np.int32)


def addSeam(img, path):
    r, c, _ = img.shape
    newImg = np.zeros((r, c + 1, 3))
    for i, j in enumerate(path):
        newImg[i, 0:j, :] = img[i, 0:j, :]
        newImg[i, j + 1 : c + 1, :] = img[i, j:c, :]
        newImg[i, j, :] = (
            img[i, j, :].astype(np.int32) + img[i, j + 1, :].astype(np.int32)
        ) // 2
    return newImg.astype(np.int32)


def rgbToGrey(arr):
    greyVal = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
    return np.round(greyVal).astype(np.int32)


# 读取图像
img = Image.open("input.jpg")
img = np.array(img)
dataColor = np.asarray(img, dtype="int32")

# 输入要删除或增加的像素数
n = int(input("Enter number of pixels to be removed: "))

# 将图像转换为灰度图像


for i in range(n):
    greyImg = rgbToGrey(dataColor)
    edgeImg = getEdge(greyImg)
    cost = findCostArr(edgeImg)
    path = findSeam(cost)
    dataColor = removeSeam(dataColor, path)

# 保存图像
Image.fromarray(dataColor.astype(np.uint8)).save("output.png")
