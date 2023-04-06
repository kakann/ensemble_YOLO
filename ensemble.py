from run_model import ObjectDetectorEnsemble


def main():

    conf_range = [0.5, 0.6,0.7, 0.8, 0.9, 0.95, 0.999]
    iou_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    print("hrj")
    #TODO
    #add so that a list of IoUs and confs can be passed for each model.
    #also seperate iou, conf for the models and iou conf for the ensemble
    ensemble= ObjectDetectorEnsemble(models= [ "yolov8x_640.pt", "yolov5x6_1280.pt"], ensemble_method="nms") # 

    ensemble.predict("test_input")

main()