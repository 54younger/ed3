from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from PIL import ImageFilter

class SeamCarvingGUI:
    def __init__(self, master, image_path):
        self.master = master
        self.master.title("Seam Carving")
        self.master.geometry("800x600")

        # 读取图像
        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)
        self.label = Label(master, image=self.photo)
        self.label.pack()

        # 创建调整大小按钮
        self.shrink_button = Button(master, text="Shrink", command=self.shrink_image)
        self.shrink_button.pack(side=LEFT)
        self.grow_button = Button(master, text="Grow", command=self.grow_image)
        self.grow_button.pack(side=LEFT)

        # 初始化变量
        self.width, self.height = self.image.size

    def energy_map(self, image):
        """
        计算图像每个像素的能量值
        """
        gray = image.convert('L')
        gradient_x = np.array(gray.filter(ImageFilter.Kernel((3, 3), [-1, 0, 1, -2, 0, 2, -1, 0, 1])))
        gradient_y = np.array(gray.filter(ImageFilter.Kernel((3, 3), [-1, -2, -1, 0, 0, 0, 1, 2, 1])))
        energy = np.abs(gradient_x) + np.abs(gradient_y)
        return energy

    def find_seam(self, cost):
        """
        找到能量值最小的路径
        """
        r, c = cost.shape

        path = []
        j = cost[0].argmin()
        path.append(j)

        for i in range(r - 1):
            c1, c2 = max(j - 1, 0), min(c, j + 2)
            j = max(j - 1, 0) + cost[i + 1, c1:c2].argmin()
            path.append(j)

        return path
    def remove_seam(self, img, path):
        """
        删除能量值最小的路径
        """
        r, c, _ = img.shape
        newImg = np.zeros((r, c, 3))
        for i, j in enumerate(path):
            newImg[i, 0:j, :] = img[i, 0:j, :]
            newImg[i, j : c - 1, :] = img[i, j + 1 : c, :]
        return newImg[:, :-1, :].astype(np.int32)

    def add_seam(self, img, path):
        """
        增加能量值最小的路径
        """
        r, c, _ = img.shape
        newImg = np.zeros((r, c + 1, 3))
        for i, j in enumerate(path):
            newImg[i, 0:j, :] = img[i, 0:j, :]
            newImg[i, j + 1 : c + 1, :] = img[i, j:c, :]
            newImg[i, j, :] = (
                img[i, j, :].astype(np.int32) + img[i, j + 1, :].astype(np.int32)
            ) // 2
        return newImg.astype(np.int32)

    def shrink_image(self):
        """
        缩小图像
        """
        energy = self.energy_map(self.image)
        seam = self.find_seam(energy)
        self.image = self.remove_seam(self.image, seam)
        self.photo = ImageTk.PhotoImage(self.image)
        self.label.configure(image=self.photo)
        self.width -= 1

    def grow_image(self):
        """
        增大图像
        """
        energy = self.energy_map(self.image)
        seam = self.find_seam(energy)
        self.image = self.add_seam(self.image, seam)
        self.photo = ImageTk.PhotoImage(self.image)
        self.label.configure(image=self.photo)
        self.width += 1

if __name__ == '__main__':
    root = Tk()
    app = SeamCarvingGUI(root, 'input.png')
    root.mainloop()
