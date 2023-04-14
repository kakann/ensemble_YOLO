from run_model import ObjectDetectorEnsemble


def main():

    iou_range = [0.5, 0.6,0.7, 0.8, 0.9, 0.95, 0.999]
    conf_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    #print("hrj")
    ensemble= ObjectDetectorEnsemble(models= [ "yolov8x_640.pt", "yolov5x6_1280.pt"], conf=0.4, ensemble_methods=["nms", "wbf"], iou=0.6) # 

    ensemble.predict(img_folder="test_input", gt_folder="test_gt", predict_folders=["test_pred_folder","test_pred_folder_copy"])
    #Maybe add f1/conf plot and similar plots options. Could be rough, but might be worth it=?
    ensemble.compare_models()

    

main()


yolov8_conf_range= [0.1, 0,2, 0,3, 0.4, 0.5]
yolov8_iou_range = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999]

yolov5_iou_range = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
yolov5_conf_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#save_txt should be true