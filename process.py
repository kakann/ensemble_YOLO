import os
import csv
from PIL import Image

def find_folders(root, s, folders_of_interest):
    found_folders = []
    for path, dirs, _ in os.walk(root):
        for d in dirs:
            if s in d and any(folder_of_interest in d for folder_of_interest in folders_of_interest):
                found_folders.append(os.path.join(path, d))
    return found_folders

def yolo_to_csv(output_csv_name, input_folder_mapping, subfolders_of_interest):
    output_folder = output_csv_name.replace(':', '_').replace('.', '_')
    os.makedirs(output_folder, exist_ok=True)

    for subfolder_of_interest in subfolders_of_interest:
        with open(os.path.join(output_folder, f"{subfolder_of_interest}.csv"), 'w', newline='') as csvfile:
            for input_folder, image_directory in input_folder_mapping.items():
                
                subfolder_path = os.path.join(input_folder, subfolder_of_interest)
                print(subfolder_path)
                yolo_files = [f for f in os.listdir(subfolder_path) if f.endswith('.txt')]
                #print(f"length of mapping {len(yolo_files)}")
                for yolo_file in yolo_files:
                    with open(os.path.join(subfolder_path, yolo_file), 'r') as file:
                        boxes = []
                        img_filename = yolo_file.replace('.txt', '.jpg')
                        img_path = os.path.join(image_directory, img_filename)
                        img = Image.open(img_path)
                        width, height = img.size
                        scores = []
                        for i, line in enumerate(file):
                            label, x, y, w, h, s = map(float, line.strip().split(' '))
                            label += 1
                            x1 = int((x - w / 2) * width)
                            y1 = int((y - h / 2) * height)
                            x2 = int((x + w / 2) * width)
                            y2 = int((y + h / 2) * height)
                            boxes.append(f"{int(label)} {x1} {y1} {x2} {y2}")
                            scores.append((s, i))
                        if len(boxes) > 5:
                            boxes_new = []
                            scores.sort(key=lambda x: x[0], reverse=True)
                            for (_, i) in scores[:5]:
                                boxes_new.append(boxes[i])
                            boxes = boxes_new
                            
                            
                        


                        if boxes:
                            csvfile.write(f'{img_filename},{boxes[0]} {" ".join(boxes[1:])}\n')

s = "IoU:0.3_Conf:0.5"
countries = ["China_MotorBike", "Norway", "Japan", "Czech", "United_States", "India"]
country_combinations = [["Japan"], ["India"], ["Norway"], ["United_States"], countries]

output_names = ["Japan", "India", "Norway", "United_States", "all"]
subfolders_of_interest = ['nms', 'nmw', 'soft-nms', 'wbf']

for countriess, name in zip(country_combinations, output_names):
    input_folders = find_folders("runs", s, countriess)
    input_folders.sort()
    countriess.sort()
    input_folder_mapping = {input_folder: f"/home/martin/Desktop/RDD_proj/datasets/RDD2022/{country}/test/images" for input_folder, country in zip(input_folders, countriess)}
    print(input_folder_mapping)
    yolo_to_csv(name + s, input_folder_mapping, subfolders_of_interest)

