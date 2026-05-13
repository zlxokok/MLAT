import cv2
import numpy as np
from pathlib import Path
import csv


def count_unique_nonzero_values(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"无法读取图片: {image_path}")
        return 0

    non_zero_pixels = img[img != 0]

    if len(non_zero_pixels) == 0:
        return 0

    unique_values = np.unique(non_zero_pixels)
    return len(unique_values)


def process_images_folder_simple(folder_path, output_csv="results.csv"):
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    folder_path = Path(folder_path)

    if not folder_path.exists():
        print(f"文件夹不存在: {folder_path}")
        return

    results = []

    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            print(f"正在处理: {file_path.name}")

            unique_count = count_unique_nonzero_values(str(file_path))
            result_text = f"{unique_count} nuclei"

            results.append({
                'Image': file_path.name,
                'Description': result_text
            })

    if results:
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['Image', 'Description']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"\n✅ 分析完成！结果已保存到: {output_csv}")
        print(f"📊 共处理了 {len(results)} 张图片")

        # 显示结果
        print("\n结果:")
        for result in results:
            print(f"{result['Image']}: {result['Description']}")
    else:
        print("❌ 未找到支持的图片文件")


# 使用示例
if __name__ == "__main__":
    folder_to_analyze = r"C:\Users\zlx\Downloads\archive (10)\cpm17\train\labelcol\\"  # 请修改为你的图片文件夹路径
    process_images_folder_simple(folder_to_analyze, r"C:\Users\zlx\Downloads\archive (10)\cpm17\train\Train_xlsx.csv")