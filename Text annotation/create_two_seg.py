import os
import cv2
import numpy as np
import pandas as pd

def process_polyp_images(folder_path):
    results = []
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for image_file in image_files:
        print(f"处理图片: {image_file}")
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_file}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        height, width = binary.shape
        left_half = binary[:, :width // 2]
        right_half = binary[:, width // 2:]

        left_white_pixels = np.sum(left_half == 255)
        right_white_pixels = np.sum(right_half == 255)

        lateral_result = ""
        if left_white_pixels > 100 and right_white_pixels > 100:
            lateral_result = "bilateral polyp"
        elif left_white_pixels > 100:
            lateral_result = "left polyp"
        elif right_white_pixels > 100:
            lateral_result = "right polyp"
        else:
            lateral_result = "no polyp detected"

        polyp_count = 0
        if lateral_result != "no polyp detected":
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            min_area = 100
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    polyp_count += 1

        polyp_count_result = f"{polyp_count} polyps"

        vertical_third = height // 3
        horizontal_half = width // 2

        regions = {
            "upper left": binary[:vertical_third, :horizontal_half],
            "middle left": binary[vertical_third:2 * vertical_third, :horizontal_half],
            "lower left": binary[2 * vertical_third:, :horizontal_half],
            "upper right": binary[:vertical_third, horizontal_half:],
            "middle right": binary[vertical_third:2 * vertical_third, horizontal_half:],
            "lower right": binary[2 * vertical_third:, horizontal_half:]
        }

        detected_regions = []
        region_threshold = 50

        for region_name, region_img in regions.items():
            white_pixels = np.sum(region_img == 255)
            if white_pixels > region_threshold:
                detected_regions.append(region_name)

        position_result = ""
        if detected_regions:
            left_regions = [r for r in detected_regions if 'left' in r]
            right_regions = [r for r in detected_regions if 'right' in r]

            if len(left_regions) == 3 and len(right_regions) == 3:
                position_result = "all regions"
            elif len(left_regions) == 3:
                position_result = "left side"
            elif len(right_regions) == 3:
                position_result = "right side"
            else:
                if left_regions and right_regions:
                    position_result = " and ".join(detected_regions)
                elif left_regions:
                    position_result = " and ".join(left_regions)
                else:
                    position_result = " and ".join(right_regions)
        else:
            position_result = "no specific location"

        if lateral_result == "no polyp detected":
            final_result = "no polyp detected"
        else:
            final_result = f"{lateral_result}, {polyp_count_result}, {position_result}"

        results.append({
            'Image': image_file,
            'Description': final_result
        })
        print(f"处理完成: {image_file} -> {final_result}")

    return results

def save_results_to_csv(results, output_file=r"C:\Users\zlx\Desktop\dataset\Kvasir-SEG\train\Train_text.csv"):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, header=['Image', 'Description'])
    print(f"结果已保存到: {output_file}")

folder_path = r"C:\Users\zlx\Desktop\dataset\Kvasir-SEG\train\labelcol\\"

if not os.path.exists(folder_path):
    print(f"文件夹不存在: {folder_path}")
    print("请修改代码中的folder_path变量为正确的图片文件夹路径")
else:
    results = process_polyp_images(folder_path)
    if results:
        save_results_to_csv(results)
        print(f"成功处理 {len(results)} 张图片。")
    else:
        print("未找到以L开头的图片或处理失败。")