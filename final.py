from PIL import Image
import numpy as np

KERNEL = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


# 转置矩阵
def GreyData(RGBimg):
    gray = RGBimg.convert("L")
    return np.round(gray).astype(np.int32)


def pre_process():
    img = Image.open("input.jpg")
    img = np.array(img)
    ImgData = np.asarray(img, dtype="int32")
    return ImgData


def Convolve(img, kernel):
    new_img = np.fft.rfft2(img)
    Con_kernel = np.fft.rfft2(kernel, img.shape)
    return np.fft.irfft2(new_img * Con_kernel)


def Get_Edge(GreyImg):
    Xedge = Convolve(GreyImg, KERNEL)
    Yedge = Convolve(GreyImg, KERNEL)
    EdgeImg = np.sqrt(np.square(Xedge) + np.square(Yedge))
    return EdgeImg


def Get_energy(EdgeImg):
    height, width = EdgeImg.shape
    energy = np.zeros(EdgeImg.shape)
    energy[height - 1, :] = EdgeImg[height - 1, :]
    for i in range(height - 2, -1, -1):
        for j in range(width):
            wid1, wid2 = max(j - 1, 0), min(width, j + 2)
            energy[i][j] = EdgeImg[i][j] + energy[i + 1, wid1:wid2].min()
    return energy

def Get_path(energy):
    path=[]
    height, width = energy.shape
    index=energy[0].argmin()
    path.append(index)

    for i in range(height-1):
        wid1,wid2=()
        max(index - 1, 0) + energy[i + 1, wid1:wid2].argmin()
