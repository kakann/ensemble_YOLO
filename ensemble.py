from run_model import ObjectDetectorEnsemble
import time
import os
import glob
from PIL import Image

def main():

    iou_range = [0.3]
    conf_range = [0.5]
    
    #print("hrj")
    #missed China motorbike "Japan", "India", "Norway", "United_States", "Czech", 
    countries = [ "China_MotorBike", "Japan", "India", "Norway", "United_States", "Czech"]
    for iou in iou_range:
        for conf in conf_range:
            start_time_ns = time.perf_counter_ns()
            ensemble= ObjectDetectorEnsemble(iou=iou, conf=conf, models= [ "yolov8x_640.pt", "yolov8x_1280.pt", "yolov8x_1600.pt", "yolov5x6_1280.pt"], ensemble_methods=["nms", "wbf", "soft-nms", "nmw"], offline=True) # 
            for country in countries:
                
                img_folder= f"/home/martin/Desktop/RDD_proj/datasets/RDD2022/{country}/test/images"
                folder0 = f"/home/martin/Desktop/pred2/640/v2yolov8x_640.pt_{country}/labels"
                folder1 = f"/home/martin/Desktop/pred2/1280/v2yolov8x_1280.pt_{country}/labels"
                
                folder2 = f"/home/martin/Desktop/pred2/1600/v2yolov8x_1600.pt_{country}/labels"
                folder3 = f"/home/martin/Desktop/pred2/yolov5_1280/v2yolov5x6_1280.pt_{country}/labels"
                ensemble.predict(img_folder=img_folder, predict_folders=[folder0, folder1, folder2, folder3], out_folder=country)
                
            #Maybe add f1/conf plot and similar plots options. Could be rough, but might be worth it=?
            #ensemble.compare_models()
            end_time_ns = time.perf_counter_ns()
            elapsed_time_ns = end_time_ns - start_time_ns
            elapsed_time_s = elapsed_time_ns / 1e9
            print(f"Time taken: {elapsed_time_s}")

    # Usage
    root_folder = "runs"
    unique_string = "iou_0.3_conf_0.3"
    country_img_dirs = {
    "Norway": "/home/martin/Desktop/RDD_proj/datasets/RDD2022/Norway/test/images",
    "United_States": "/home/martin/Desktop/RDD_proj/datasets/RDD2022/United_States/test/images",
    "India": "/home/martin/Desktop/RDD_proj/datasets/RDD2022/India/test/images",
    "Japan": "/home/martin/Desktop/RDD_proj/datasets/RDD2022/Japan/test/images",
    "Czech": "/home/martin/Desktop/RDD_proj/datasets/RDD2022/Czech/test/images",
    "China_Motorbike": "/home/martin/Desktop/RDD_proj/datasets/RDD2022/China_MotorBike/test/images",
    }
    process_folders(root_folder, unique_string, country_img_dirs)


def yolo_norm_to_abs_xyxy(norm_bbox, img_width, img_height):
    x_center, y_center, width, height = norm_bbox
    xmin = (x_center - width / 2) * img_width
    ymin = (y_center - height / 2) * img_height
    xmax = (x_center + width / 2) * img_width
    ymax = (y_center + height / 2) * img_height
    return xmin, ymin, xmax, ymax

def process_folders(root_folder, unique_string, country_img_dirs, countries=['Norway', 'United_States', 'India', 'Japan']):
    # Find directories containing the unique_string
    selected_directories = [d for d in os.listdir(root_folder) if unique_string in d]

    # Create output files for each country and for all countries combined
    output_files = {}
    for country in countries:
        output_files[country] = open(f"{country}_output.txt", "w")
    output_files["All"] = open("All_output.txt", "w")

    # Iterate through selected directories
    for directory in selected_directories:
        # Identify the country for the current directory
        country = None
        for c in countries:
            if c in directory:
                country = c
                break

        # Read files in the current directory
        files = glob.glob(os.path.join(root_folder, directory, "*.txt"))
        for file in files:
            file_name, _ = os.path.splitext(os.path.basename(file))
            img_path = os.path.join(country_img_dirs[country], f"{file_name}.jpg")
            img = Image.open(img_path)
            img_width, img_height = img.size

            with open(file, 'r') as f:
                predictions = [line.strip() for line in f.readlines()]

            abs_predictions = []
            for pred in predictions:
                label, x_center, y_center, width, height = map(float, pred.split(' '))
                abs_bbox = yolo_norm_to_abs_xyxy((x_center, y_center, width, height), img_width, img_height)
                abs_predictions.append(f"{label} {' '.join(map(str, abs_bbox))}")

            # Write predictions to the output file for the identified country and the combined file
            line = f"{file_name}.jpg, " + " ".join(abs_predictions) + "\n"
            if country:
                output_files[country].write(line)
            output_files["All"].write(line)

    # Close output files
    for file in output_files.values():
        file.close()

    

main()

#save_txt should be true