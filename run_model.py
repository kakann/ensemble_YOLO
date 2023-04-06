import os
import cv2
import numpy as np
from ensemble_boxes import *
import subprocess

from ultralytics import YOLO
import yolov5
import time
from yolov5.utils.metrics import ConfusionMatrix
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator

class Prediction:
    def __init__(self, img, bbox, score, label ):
        self.img=img
        self.bbox=bbox
        self.score = score
        self.label = label

class ImageObject:
    def __init__(self, imgpath, imgsz) -> None:
        pass

class Model:
    def __init__(self, model_name, modelv) -> None:
        self.model_name=model_name
        self.modelv=modelv

class ModelResult:
    def __init__(self, predList, model) -> None:
        pass
        
        


class ObjectDetectorEnsemble:
    def __init__(self, models, ensemble_method='mean', conf=0.4, iou=0.9, tta=True):
        self.models = []
        self.ensemble_method = ensemble_method
        self.conf = conf
        self.iou = iou
        self.tta = tta
        self.model_names = []
        for model in models:
            self.model_names.append(model.split(".")[0])
            #print(model)
        
        print(self.model_names)
        for weights in models:
            try: 
               yolov5.load(weights)
               self.models.append((weights, "yolov5"))
               print("v5")
            except:
                print("v8")
                YOLO(weights)
                self.models.append((weights, "yolov8"))


    
    

    def predict(self, img_folder, gt_folder=None):
        # Load the image paths in the folder
        img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
        #gt_paths = [os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith('.txt') or f.endswith('.xml')]

        #list of lists where each list contains all predictions for each model
        model_preds = []
        boxes_list, scores_list, labels_list, img_shapes_list= [], [], [], []
        for img in img_paths:
            img_shapes_list.append(cv2.imread(img).shape[:2])
        
        print(img_shapes_list)
        for (model, modelv), model_name in zip(self.models, self.model_names):
            # Make a prediction with the current model
            raw_preds = []

            if modelv == "yolov8":
                print(model)
                mod = YOLO(model)
                raw_preds = mod(img_paths, augment=self.tta, conf=self.conf, iou=self.iou)
            else:
                mod = yolov5.load(model)
                mod.conf = self.conf
                mod.iou = 0.6
                #mod.imgsz= 1280
                raw_preds = mod(img_paths, augment=self.tta) # , conf=self.conf, iou=self.iou

            
            # Add the model predictions to the list
            boxes_mod = []
            scores_mod = []
            labels_mod = []
            
            if modelv == "yolov5":
                for i in range(0, len(raw_preds)):
                    #print(pred)
                    boxes_mod.append(raw_preds.pred[i][:, :4].cpu().numpy()) # x1, y1, x2, y2
                    scores_mod.append(raw_preds.pred[i][:, 4].cpu().numpy())
                    labels_mod.append(raw_preds.pred[i][:, 5].cpu().numpy())
            if modelv == "yolov8":
                for result in raw_preds:
                    result = result.boxes.boxes
                    boxes_mod.append(result[:, :4].cpu().numpy()) # x1, y1, x2, y2
                    scores_mod.append(result[:, 4].cpu().numpy())
                    labels_mod.append(result[:, 5].cpu().numpy())
                print("v8")

            #print(boxes_mod)
            
            boxes_list.append(boxes_mod)
            scores_list.append(scores_mod)
            labels_list.append(labels_mod)
        
        j = 0
        #for each first img predictions from each model, should iterate once for each i images.
        for model_predictions_boxes, model_predictions_scores, model_predictions_labels in zip(zip(*boxes_list), zip(*scores_list), zip(*labels_list)):
            print(f"Doing {self.ensemble_method} on image {j}")
            bboxes= []
            scores= []
            labels=[]


            for i in range(len(model_predictions_boxes)):
                width = img_shapes_list[j][1]
                height = img_shapes_list[j][0]

                boxes= model_predictions_boxes[i].tolist()

                norm_boxes = [[coord / width if idx % 2 == 0 else coord / height for idx, coord in enumerate(coords)]for coords in boxes]

                bboxes += [norm_boxes]
                scores += [model_predictions_scores[i].tolist()]
                labels += [model_predictions_labels[i].tolist()]

            j+=1
        self.pick_ensemble(bboxes, scores, labels)
            
            
    def pick_ensemble(self, bboxes, scores, labels):
        # Combine the model predictions using the ensemble method

        if self.ensemble_method == 'nms':
            boxes, scores, labels = nms(bboxes, scores, labels, iou_thr=self.iou)
            combined_preds = np.column_stack((boxes, scores, labels))

        elif self.ensemble_method == 'soft_nms':
            boxes, scores, labels = soft_nms(bboxes, scores, labels, method=2, iou_thr=self.iou)
            combined_preds = np.column_stack((boxes, scores, labels))
        elif self.ensemble_method == 'nmw':

            boxes, scores, labels = non_maximum_weighted(bboxes, scores, labels, iou_thr=self.iou)
            combined_preds = np.column_stack((boxes, scores, labels))
        elif self.ensemble_method == 'wbf':

            boxes, scores, labels = weighted_boxes_fusion(bboxes, scores, labels, iou_thr=self.iou)
            combined_preds = np.column_stack((boxes, scores, labels))
        elif self.ensemble_method == "OBB":
            #bboxes, scores, labels = [], [], []
            subprocess.run(['python', 'program.py'])
        return np.column_stack((boxes, scores, labels))
        

    #quickfixed, worked before but might need to take input format into close concideration
    def box_imgs(model_name, bboxes, scores, labels, output_folder, img_paths):
        import pathlib
        pathlib.Path(f"{output_folder}/{model_name}").mkdir(parents=True, exist_ok=True) 
        #os.mkdir(f"test_out/{model_name}")
        i =0
        for img in img_paths:
            img1 = cv2.imread(img)
            if img1 is None:
                print(f"Failed to read image: {img}")
                continue
            annotator = Annotator(img1)
            for bbox, score, label in zip(bboxes, scores, labels):
                #print(bbox)
                #print(f"current place {j}")
                annotator.box_label(box=bbox, label=f"{label} {score}", )
            
            cv2.imshow('image',img1)
            cv2.waitKey(1000)
            cv2.imwrite(f"{output_folder}/{model_name}/{i}.jpg", img1)
            i+=1     


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