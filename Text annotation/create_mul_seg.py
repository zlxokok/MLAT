import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import glob

def find_connected_components(img_array, target_rgb):
    if len(img_array.shape) == 3:
        mask = np.all(img_array == target_rgb, axis=-1).astype(np.uint8) * 255
    else:
        mask = ((img_array == target_rgb) * 255).astype(np.uint8)
    if not np.any(mask):
        return 0
    num_labels, labels = cv2.connectedComponents(mask)
    return num_labels - 1

def analyze_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            pil_img = Image.open(image_path)
            img = np.array(pil_img)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"无法读取图片 {image_path}: {e}")
        return None

    rgb_mappings = {
        55: "SRF",
        105: "PED",
        155: "IRF",
        205: "SHRM",
        255: "IS/OS"
    }

    descriptions = []
    for rgb_value, label in rgb_mappings.items():
        num_regions = find_connected_components(img, rgb_value)
        if num_regions > 0:
            descriptions.append(f"{num_regions} {label}")

    return ", ".join(descriptions) if descriptions else "No findings"

def main():
    image_folder = input("请输入图片文件夹路径: ").strip()
    if not os.path.exists(image_folder):
        print("文件夹不存在！")
        return

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

    image_files = []
    found_files = set()

    for extension in image_extensions:
        lower_files = glob.glob(os.path.join(image_folder, extension))
        upper_files = glob.glob(os.path.join(image_folder, extension.upper()))
        for file_path in lower_files + upper_files:
            normalized_path = os.path.normpath(file_path)
            if normalized_path not in found_files:
                found_files.add(normalized_path)
                image_files.append(normalized_path)

    if not image_files:
        for file in os.listdir(image_folder):
            file_path = os.path.join(image_folder, file)
            file_lower = file.lower()
            if any(file_lower.endswith(ext.replace('*', '')) for ext in image_extensions):
                normalized_path = os.path.normpath(file_path)
                if normalized_path not in found_files:
                    found_files.add(normalized_path)
                    image_files.append(normalized_path)

    if not image_files:
        print("在指定文件夹中未找到图片文件")
        return

    print(f"找到 {len(image_files)} 个图片文件")

    results = []
    processed_files = set()

    for image_file in image_files:
        if image_file in processed_files:
            continue
        print(f"正在分析: {os.path.basename(image_file)}")
        description = analyze_image(image_file)
        processed_files.add(image_file)
        if description is not None:
            results.append({
                'Image': os.path.basename(image_file),
                'Description': description
            })

    if results:
        df = pd.DataFrame(results)
        output_file = r'C:\Users\zlx\Desktop\dataset\AMDSD\test\Val_text.xlsx'
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_excel(output_file, index=False)
        print(f"\n分析完成！结果已保存到: {output_file}")
        print("\n分析摘要:")
        for result in results:
            print(f"{result['Image']}: {result['Description']}")
        image_names = [result['Image'] for result in results]
        duplicates = set([x for x in image_names if image_names.count(x) > 1])
        if duplicates:
            print(f"\n警告: 发现重复的图片名: {duplicates}")
    else:
        print("没有成功分析任何图片")

if __name__ == "__main__":
    main()