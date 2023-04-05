import os
import cv2
import numpy as np
from ensemble_boxes import *
import subprocess

from ultralytics import YOLO
import yolov5

from yolov5.utils.metrics import ConfusionMatrix
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator



class ObjectDetectorEnsemble:
    def __init__(self, models, ensemble_method='mean', conf=0.3, iou=0.9, tta=True):
        #self.models = models
        self.ensemble_method = ensemble_method
        self.conf = conf
        self.iou = iou
        self.tta = tta
        for model in models:
            self.model_names.append(model.split(".")[0])
        
        for weights in models:
            try:
                self.models.append(yolov5.load(weights), "yolov5")
            except:
                self.models.append(YOLO(weights), "yolov8")


    
            

    def predict(self, img_folder, gt_folder):
        # Load the image paths in the folder
        img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
        gt_paths = [os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith('.txt') or f.endswith('.xml')]

        #list of lists where each list contains all predictions for each model
        model_preds = []
        boxes_list, scores_list, labels_list = [], [], []
        for model, modelv in self.models:
            # Make a prediction with the current model
            raw_preds = model(img_paths, augment=self.tta, conf=self.conf, iou=self.iou )
                
            # Add the model predictions to the list
            model_preds.append(raw_preds)
            print(raw_preds)
            boxes_mod = []
            scores_mod = []
            labels_mod = []
            if modelv == "yolov5":
                boxes_mod = (raw_preds[:, :4]) # x1, y1, x2, y2
                scores_mod = (raw_preds[:, 4])
                labels_mod = (raw_preds[:, 5])
            if modelv == "yolov8":
                print("v8")
            
            plot_one_box(bbox, img, label=class_label, score=score)

        
        all_model_preds = 0
        # Combine the model predictions using the ensemble method
        if self.ensemble_method == 'mean':
            combined_preds = np.mean(all_model_preds, axis=0)
        elif self.ensemble_method == 'max':
            combined_preds = np.maximum.reduce(all_model_preds)
        elif self.ensemble_method == 'nms':
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=self.iou)
            combined_preds = np.column_stack((boxes, scores, labels))

        elif self.ensemble_method == 'soft_nms':
            boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, method=2, iou_thr=self.iou)
            combined_preds = np.column_stack((boxes, scores, labels))
        elif self.ensemble_method == 'nmw':

            boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, iou_thr=self.iou)
            combined_preds = np.column_stack((boxes, scores, labels))
        elif self.ensemble_method == 'wbf':

            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=self.iou)
            combined_preds = np.column_stack((boxes, scores, labels))
        elif self.ensemble_method == "OBB":
            #boxes_list, scores_list, labels_list = [], [], []
            subprocess.run(['python', 'program.py'])
        return np.column_stack((boxes, scores, labels))
            
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

            


    def convert_files(folder_path, format):
    # Check if the format is valid
        assert format in ["yolo", "voc"], "Invalid format"

        # Counters for txt and xml files
        num_txt_files = 0
        num_xml_files = 0

        # List to store paths to txt and xml files
        txt_file_paths = []
        xml_file_paths = []

        # Walk through the folder and its subfolders
        for root, dirs, files in os.walk(folder_path):
            # Iterate through files in the current folder
            for file in files:
                # Check if the file has a txt or xml extension
                if file.endswith(".txt"):
                    num_txt_files += 1
                    txt_file_paths.append(os.path.join(root, file))
                elif file.endswith(".xml"):
                    num_xml_files += 1
                    xml_file_paths.append(os.path.join(root, file))

        # Print the summary
        print(f"Number of txt files found: {num_txt_files}")
        print(f"Number of xml files found: {num_xml_files}")

        #if format == "yolo":
        #    for xml in xml_file_paths: