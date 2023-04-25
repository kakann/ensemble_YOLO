from run_model import ObjectDetectorEnsemble
import time

def main():

    iou_range = [0.1, 0.3, 0.4, 0.6, 0.8]
    conf_range = [0.3, 0.4, 0.5, 0.6]
    
    #print("hrj")
    
    countries = ["India", "Japan", "Norway", "United_States", "Czech"]
    start_time_ns = time.perf_counter_ns()
    ensemble= ObjectDetectorEnsemble(iou=iou_range[2], conf=conf_range[0], models= [ "yolov8x_640.pt", "yolov8x_1280.pt", "yolov8x_1600.pt", "yolov5x6_1280.pt"], ensemble_methods=["nms", "wbf", "soft-nms", "nmw"]) # 
    for country in countries:
        
        img_folder= f"/home/martin/Desktop/RDD_proj/datasets/RDD2022/{country}/test/images"
        folder0 = f"/home/martin/Desktop/Predictions/1280/yolov8x_1280.pt_{country}/labels"
        folder1 = f"/home/martin/Desktop/Predictions/640/yolov8x_640.pt_{country}/labels"
        folder2 = f"/home/martin/Desktop/Predictions/1600/yolov8x_1600.pt_{country}/labels"
        folder3 = f"/home/martin/Desktop/Predictions/yolov5_1280/yolov5x6_1280.pt_{country}/labels"
        ensemble.predict(img_folder=img_folder, predict_folders=[folder0, folder1, folder2, folder3], out_folder=country)
        
    #Maybe add f1/conf plot and similar plots options. Could be rough, but might be worth it=?
    #ensemble.compare_models()
    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_s = elapsed_time_ns / 1e9
    print(f"Time taken: {elapsed_time_s}")

    

main()


yolov8_conf_range= [0.1, 0,2, 0,3, 0.4, 0.5]
yolov8_iou_range = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999]

yolov5_iou_range = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
yolov5_conf_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#save_txt should be true