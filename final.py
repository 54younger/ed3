from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from poprogress import simple_progress

KERNEL = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
KERNEL_T = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
last_energy_min = []


def ChangeIntoGrey(RGBimg):
    gray = np.dot(RGBimg[..., :3], [0.33, 0.33, 0.33])
    return np.round(gray).astype(np.int32)


def pre_process(how):
    img = Image.open("input.jpg")
    img = np.array(img)
    if how == "v":  # vertical
        img = np.transpose(img, (1, 0, 2))
    ImgData = np.asarray(img, dtype="int32")
    global last_energy_min
    last_energy_min = np.zeros(ImgData.shape[0], dtype=int)
    return ImgData


def Convolve(img, kernel):
    new_img = np.fft.rfft2(img)
    Con_kernel = np.fft.rfft2(kernel, img.shape)
    return np.fft.irfft2(new_img * Con_kernel)


def Get_Edge(GreyImg):
    Xedge = Convolve(GreyImg, KERNEL)
    Yedge = Convolve(GreyImg, KERNEL_T)
    EdgeImg = np.sqrt(np.square(Xedge) + np.square(Yedge))
    return EdgeImg


def Get_energy(EdgeImg):
    height, width = EdgeImg.shape
    energy = np.zeros(EdgeImg.shape)
    energy[height - 1, :] = EdgeImg[height - 1, :]
    energy[height - 1, (last_energy_min[height - 1]-1):(last_energy_min[height - 1]+1)] = 10000000
    for i in range(height - 2, -1, -1):
        for j in range(width):
            energy[i][j] = (
                EdgeImg[i][j] + energy[i + 1, max(j - 1, 0) : min(width, j + 2)].min()
            )
            if last_energy_min[i] - 1 <= j <= last_energy_min[i] + 1:
                energy[i][j] = 10000000

    return energy


def Get_seam(energy):
    seam = []
    global last_energy_min

    height, width = energy.shape
    index = energy[0].argmin()
    last_energy_min[0] = index
    seam.append(index)

    for i in range(height - 1):
        index = (
            max(index - 1, 0)
            + energy[i + 1, max(index - 1, 0) : min(width, index + 2)].argmin()
        )
        last_energy_min[i + 1] = index
        seam.append(index)

    return seam


def CreateNewImg(img, path, act):
    height, width, _ = img.shape

    if act == 1:  # add
        newImg = np.zeros((height, width + 1, 3))
        for i, j in enumerate(path):
            newImg[i, 0:j, :] = img[i, 0:j, :]
            newImg[i, j + 1 : width + 1, :] = img[i, j:width, :]
            newImg[i, j, :] = (
                img[i, j, :].astype(np.int32) + img[i, j + 1, :].astype(np.int32)
            ) // 2
        return newImg.astype(np.int32)
    else:  # delete
        newImg = np.zeros((height, width, 3))
        for i, j in enumerate(path):
            newImg[i, 0:j, :] = img[i, 0:j, :]
            newImg[i, j : width - 1, :] = img[i, j + 1 : width, :]
            newImg[i, j, :] = (
                img[i, j, :].astype(np.int32) + img[i, j + 1, :].astype(np.int32)
            ) // 2
        return newImg[:, :-1, :].astype(np.int32)


how = input("The direction process(h/v):")
ImgData = pre_process(how)
originalImg = ImgData.copy()
act = int(input("Add(1) or Delete(0):"))

n = int(input("Enter number of pixels to be removed: "))

for i in simple_progress(range(n)):
    GreyImg = ChangeIntoGrey(ImgData)
    EdgeImg = Get_Edge(GreyImg)
    energy = Get_energy(EdgeImg)
    seam = Get_seam(energy)
    ImgData = CreateNewImg(ImgData, seam, act)

if how == "v":
    ImgData = np.transpose(ImgData, (1, 0, 2))
    originalImg = np.transpose(originalImg, (1, 0, 2))

newImg = Image.fromarray(ImgData.astype(np.uint8))

# 对比更改前后图像
fig, axs = plt.subplots(1, 2)
axs[0].imshow(originalImg)
axs[0].set_title("Original Image")
axs[1].imshow(newImg)
axs[1].set_title("Modified Image")
plt.show()
# 储存图像
if act == 1:
    name = f"output-{how}ADD{n}pixels.png"
else:
    name = f"output-{how}Deleten{n}pixels.png"
newImg.save(name)
