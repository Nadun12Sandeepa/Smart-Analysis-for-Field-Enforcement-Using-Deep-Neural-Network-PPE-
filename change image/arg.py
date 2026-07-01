import os
import cv2
import albumentations as A
from tqdm import tqdm

# ===================== CONFIG =====================
input_folder = "C:\\Users\\DELL\\Desktop\\resized"     # Your original images folder
output_folder = "C:\\Users\\DELL\\Desktop\\augmented_images"  # Where augmented images will be saved
num_augmented_per_image = 5       # How many augmentations per image
image_size = 640                  # Output image size (640x640)
# ===================================================

os.makedirs(output_folder, exist_ok=True)

# Define augmentation pipeline
transform = A.Compose([
    A.Resize(image_size, image_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.RGBShift(p=0.3),
    A.GaussNoise(p=0.2),
    A.MotionBlur(p=0.2),
    A.GridDistortion(p=0.3),
    A.ElasticTransform(p=0.1),
    A.CoarseDropout(max_holes=8, max_height=64, max_width=64, fill_value=0, p=0.5)
])

# Process all images
for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(num_augmented_per_image):
            augmented = transform(image=image)
            aug_image = augmented["image"]
            aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)

            out_name = f"{os.path.splitext(filename)[0]}_aug{i+1}.jpg"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, aug_image)

print("✅ Augmentation complete. Images saved to:", output_folder)
