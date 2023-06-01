import cv2
import numpy as np
from functools import lru_cache
from multiprocessing import Pool
import logging
import argparse


class SeamCarver:
    def __init__(self, image):
        self.image = image
        self.energy = None
        self.dp = None
        self.seam = None

    @staticmethod
    def _energy_map(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy = np.abs(gradient_x) + np.abs(gradient_y)
        return energy

    @staticmethod
    def _find_seam(dp, energy):
        m, n = energy.shape
        seam = np.zeros(m, dtype=np.int32)
        seam[-1] = np.argmin(dp[-1])
        for i in range(m - 2, -1, -1):
            j = seam[m - i - 1]
            if j == 0:
                seam[m - i - 2] = np.argmin(dp[i][j : j + 2]) + j
            elif j == n - 1:
                seam[m - i - 2] = np.argmin(dp[i][j - 1 : j + 1]) + j - 1
            else:
                seam[m - i - 2] = np.argmin(dp[i][j - 1 : j + 2]) + j - 1
        return seam

    @lru_cache(maxsize=None)
    def _dp(self, i, j):
        if i == 0:
            return self.energy[0][j]
        if j == 0:
            return self.energy[i][j] + min(self._dp(i - 1, j), self._dp(i - 1, j + 1))
        if j == self.energy.shape[1] - 1:
            return self.energy[i][j] + min(self._dp(i - 1, j - 1), self._dp(i - 1, j))
        return self.energy[i][j] + min(
            self._dp(i - 1, j - 1), self._dp(i - 1, j), self._dp(i - 1, j + 1)
        )

    def _compute_energy(self):
        self.energy = self._energy_map(self.image)

    def _compute_dp(self):
        self.dp = np.zeros_like(self.energy)
        for i in range(self.energy.shape[1]):
            self.dp[0][i] = self.energy[0][i]
        for i in range(1, self.energy.shape[0]):
            for j in range(self.energy.shape[1]):
                self.dp[i][j] = self._dp(i, j)

    def _compute_seam(self):
        self.seam = self._find_seam(self.dp, self.energy)

    def remove_seam(self):
        self._compute_energy()
        self._compute_dp()
        self._compute_seam()
        new_image = np.zeros((self.image.shape[0], self.image.shape[1] - 1, 3), dtype=np.uint8)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1] - 1):
                if j < self.seam[i]:
                    new_image[i][j] = self.image[i][j]
                else:
                    new_image[i][j] = self.image[i][j + 1]
        self.image = new_image

    def add_seam(self):
        self._compute_energy()
        self._compute_dp()
        self._compute_seam()
        new_image = np.zeros((self.image.shape[0], self.image.shape[1] + 1, 3), dtype=np.uint8)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1] + 1):
                if j < self.seam[i]:
                    new_image[i][j] = self.image[i][j]
                elif j == self.seam[i]:
                    if j == 0:
                        left = self.image[i][j]
                        right = self.image[i][j + 1]
                    elif j == self.image.shape[1]:
                        left = self.image[i][j - 1]
                        right = self.image[i][j]
                    else:
                        left = self.image[i][j - 1]
                        right = self.image[i][j]
                    new_pixel = (left.astype(np.int32) + right.astype(np.int32)) // 2
                    new_image[i][j] = new_pixel.astype(np.uint8)
                else:
                    new_image[i][j] = self.image[i][j - 1]
        self.image = new_image

    @classmethod
    def resize(cls, image, new_size):
        m, n = image.shape[:2]
        new_m, new_n = new_size
        if new_m < m:
            for i in range(m - new_m):
                carver = cls(image)
                carver.remove_seam()
                image = carver.image
        elif new_m > m:
            for i in range(new_m - m):
                carver = cls(image)
                carver.add_seam()
                image = carver.image
        if new_n < n:
            image = cv2.transpose(image)
            for j in range(n - new_n):
                carver = cls(image)
                carver.remove_seam()
                image = carver.image
            image = cv2.transpose(image)
        elif new_n > n:
            image = cv2.transpose(image)
            for j in range(new_n - n):
                carver = cls(image)
                carver.add_seam()
                image = carver.image
            image = cv2.transpose(image)
        return image




if __name__ == "__main__":
    image = cv2.imread("input.png")
    print("Original size:", image.shape)
    print("please input the new size:")
    new_size = tuple(map(int, input().split()))
    new_image =SeamCarver.resize(image, new_size)
    cv2.imwrite("output.jpg", new_image)
