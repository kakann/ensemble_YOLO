from yolov5.utils.metrics import ConfusionMatrix
from yolov5.utils.torch_utils import select_device
import numpy as np

# Define paths to predictions and ground truth annotations
pred_path = "path/to/predictions.txt"
gt_path = "path/to/annotations.txt"
img_size = 640
conf = 0.3
iou = 0.5
device = select_device("")

# Define the classes
classes = ["class1", "class2", "class3", ...]

def calculate_metrics(pred_path, gt_path, conf=0.3, iou= 0.8, device=select_device(), classes=["D00", "D10", "D20", "D40"]):
    # Create a ConfusionMatrix object for each class
    confusion_matrices = [ConfusionMatrix(nc=1) for _ in range(len(classes))]

    # Load the predictions and ground truth annotations
    preds = np.loadtxt(pred_path, delimiter=",")
    gts = np.loadtxt(gt_path, delimiter=",")

    # Process the batch separately for each class
    for i, class_name in enumerate(classes):
        class_preds = preds[preds[:, 5] == i]  # select the predictions for this class
        class_gts = gts[gts[:, 4] == i]  # select the ground truth annotations for this class
        confusion_matrices[i].process_batch(class_preds, class_gts, conf_thres=conf, iou_thres=iou, device=device)

    # Calculate precision, recall, and mAP separately for each class
    for i, class_name in enumerate(classes):
        class_precision, class_recall = confusion_matrices[i].precision_recall()
        class_mAP = confusion_matrices[i].average_precision()
        print(f"{class_name}: precision={class_precision:.4f}, recall={class_recall:.4f}, mAP={class_mAP:.4f}")
