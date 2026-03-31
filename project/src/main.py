#!/usr/bin/env python3
import h5py
import matplotlib.pyplot as plt

with h5py.File("../datasets/Galaxy10_DECals_NoDuplicated.h5", "r") as f:
    images = f["images"][:]
    labels = f["labels"][:]

# Mostrar algumas imagens
plt.figure(figsize=(10, 10))

for i in range(9):  # mostra 9 imagens
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(f"Label: {labels[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
