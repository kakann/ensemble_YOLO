from run_model import ObjectDetectorEnsemble


def main():

    conf_range = [0.5, 0.6,0.7, 0.8, 0.9, 0.95, 0.999]
    iou_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    ensemble= ObjectDetectorEnsemble(models= ["yolov5x6_1280.pt", "yolov8_640"], ensemble_method="nms",)