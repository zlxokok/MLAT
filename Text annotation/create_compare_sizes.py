import os
import cv2
import numpy as np
import pandas as pd
import glob

def find_connected_components(binary_image, min_area=100):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            centroid_x = centroids[i][0]
            centroid_y = centroids[i][1]
            regions.append({
                'area': area,
                'centroid': (centroid_x, centroid_y)
            })
    return regions

def process_images_in_folder(folder_path):
    results = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for extension in image_extensions:
        pattern = os.path.join(folder_path, extension)
        found_files = glob.glob(pattern)
        image_files.extend(found_files)
    image_files = list(set(image_files))

    if not image_files:
        print("在指定文件夹中未找到图片文件")
        return None

    print(f"找到 {len(image_files)} 个图片文件")

    for image_path in image_files:
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                results.append([os.path.basename(image_path), "无法读取图片"])
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            regions = find_connected_components(binary)

            if len(regions) < 2:
                print(f"图片 {os.path.basename(image_path)} 中白色连通区域数量不足2个")
                results.append([os.path.basename(image_path), "区域数量不足"])
                continue

            regions.sort(key=lambda x: x['area'], reverse=True)
            region1, region2 = regions[0], regions[1]

            if region1['centroid'][0] < region2['centroid'][0]:
                left_region, right_region = region1, region2
            else:
                left_region, right_region = region2, region1

            left_area = left_region['area']
            right_area = right_region['area']
            total_area = left_area + right_area
            area_ratio = abs(left_area - right_area) / total_area

            print(f"图片: {os.path.basename(image_path)}")
            print(f"左侧区域面积: {left_area}, 右侧区域面积: {right_area}")
            print(f"面积差异比例: {area_ratio:.2%}")

            if area_ratio <= 0.08:
                result_text = "left and right are similar"
            else:
                if left_area > right_area:
                    result_text = "left is larger"
                else:
                    result_text = "right is larger"

            print(f"结果: {result_text}")
            print("-" * 50)

            results.append([os.path.basename(image_path), result_text])

        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")
            results.append([os.path.basename(image_path), f"处理错误: {str(e)}"])

    return results

def save_results_to_excel(results, output_file="results.xlsx"):
    if not results:
        print("没有结果可保存")
        return
    df = pd.DataFrame(results, columns=['Image', 'Description'])
    try:
        df.to_excel(output_file, index=False)
        print(f"结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存Excel文件时出错: {str(e)}")

def main():
    folder_path = input("请输入图片文件夹路径: ").strip().strip('"')
    if not os.path.exists(folder_path):
        print("指定的文件夹不存在")
        return
    results = process_images_in_folder(folder_path)
    if results:
        output_filename = input("请输入输出Excel文件名（默认为results.xlsx）: ").strip()
        if not output_filename:
            output_filename = "results.xlsx"
        elif not output_filename.endswith('.xlsx'):
            output_filename += '.xlsx'
        save_results_to_excel(results, output_filename)
        successful_processing = len([r for r in results if
                                     "区域数量不足" not in r[1] and "处理错误" not in r[1] and "无法读取图片" not in r[1]])
        print(f"\n处理完成！成功处理 {successful_processing}/{len(results)} 张图片")

if __name__ == "__main__":
    main()