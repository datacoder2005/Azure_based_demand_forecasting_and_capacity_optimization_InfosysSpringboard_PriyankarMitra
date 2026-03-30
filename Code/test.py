import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================== READ IMAGE ====================
image_path = r"C:\Users\Priyankar Mitra\Documents\Downloads\images.jpg"
img = cv2.imread(image_path)

if img is None:
    print("Error loading image")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to float
img_float = img_rgb.astype(np.float64) / 255.0

# ==================== UN-SHARP MASKING ====================

print("Applying professional sharpening...")

# Step 1: Gaussian Blur
blur = cv2.GaussianBlur(img_float, (0, 0), sigmaX=2)

# Step 2: High-frequency mask
mask = img_float - blur

# Step 3: Sharpening strength (BEST RANGE: 1.5–2.5)
strength = 2.0

sharpened = img_float + strength * mask

# Clip values
sharpened = np.clip(sharpened, 0, 1)

# Convert back to uint8
sharpened_uint8 = (sharpened * 255).astype(np.uint8)

print("Sharpening complete")

# ==================== DISPLAY ====================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(np.clip(mask + 0.5, 0, 1))
axes[1].set_title("High-Frequency Mask")
axes[1].axis("off")

axes[2].imshow(sharpened_uint8)
axes[2].set_title(f"Sharpened (strength={strength})")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# Save result
cv2.imwrite("sharpened_best.jpg", cv2.cvtColor(sharpened_uint8, cv2.COLOR_RGB2BGR))

print("Saved as sharpened_best.jpg")