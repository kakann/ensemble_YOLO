import os
import cv2
import numpy as np
from ensemble_boxes import *
import subprocess
import copy

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import matplotlib.pyplot as plt
from ultralytics import YOLO
import yolov5
import time
from yolov5.utils.metrics import ConfusionMatrix
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator
from sklearn.preprocessing import label_binarize

from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score, auc

class Ensemble:
    def __init__(self, ensemblem_name, predictions) -> None:
        self.ensemblem_name = ensemblem_name
        self.predictions = predictions

class Model:
    def __init__(self, name, results=None) -> None:
        self.name = name
        self.results= results
        

class EnsembleResults:
    def __init__(self, models, ensembles) -> None:
        self.models = models
        self.ensembles = ensembles
        

class ObjectDetectorEnsemble:
    def __init__(self, models, confs, ious, ensemble_methods=["nms"], conf=0.4, iou=0.6, tta=True):
        self.models = []
        self.model_predictions = [] # list of tuples. Each tuples is bboxes, scores, labels
        self.ensemble_methods = ensemble_methods
        self.ensemble_results = [] # list of name, tuples. Each tuples is bboxes, scores, labels
        self.conf = conf
        self.iou = iou
        self.tta = tta
        self.model_names = []
        self.confs = confs
        self.ious = ious
        self.gts = []
        self.img_paths= []
        for model in models:
            self.model_names.append(model.split(".")[0])
            #print(model)
        
        print(self.model_names)
        for weights in models:
            try: 
               yolo = yolov5.load(weights)
               self.models.append((weights, "yolov5"))
               print("v5")
            except:
                print("v8")
                YOLO(weights)
                self.models.append((weights, "yolov8"))

    #Runs m models defined in self.models
    #Returns bboxes, scores, labels
    def run_models(self, img_paths):
        boxes_list, scores_list, labels_list = [], [], []
        for (model, modelv), model_name, confmod, ioumod in zip(self.models, self.model_names, self.confs, self.ious):
            # Make a prediction with the current model
            raw_preds = []

            if modelv == "yolov8":
                print(model)
                mod = YOLO(model)
                raw_preds = mod(img_paths, augment=self.tta, imgsz=640, batch=1)
            else:
                mod = yolov5.load(model)
                mod.conf = confmod
                mod.iou = ioumod
                #mod.imgsz= 1280
                raw_preds = mod(img_paths, augment=self.tta) # , conf=self.conf, iou=self.iou

            
            # Add the model predictions to the list
            boxes_mod = []
            scores_mod = []
            labels_mod = []
            
            if modelv == "yolov5": #NEED TO CHECK FOR SAME ERROR AS YOLOV8 
                for i in range(0, len(raw_preds)):
                    #print(raw_preds.xywhn[i][:, :4].cpu().numpy())
                    
                    
                    boxes_mod.append(raw_preds.xywhn[i][:, :4].cpu().numpy().tolist())
                    
                    
                    #print(norm_boxes)
                    scores_mod.append(raw_preds.pred[i][:, 4].cpu().numpy().tolist())
                    labels_mod.append(raw_preds.pred[i][:, 5].cpu().numpy().tolist())
            if modelv == "yolov8":
                for result in raw_preds:
                    result = result.boxes
                    boxes_mod.append(result.xywhn.cpu().numpy().tolist()) # x1, y1, x2, y2
                    scores_mod.append(result.conf.cpu().numpy().tolist())
                    labels_mod.append(result.cls.cpu().numpy().tolist())

            boxes_list.append(boxes_mod)
            scores_list.append(scores_mod)
            labels_list.append(labels_mod)

            self.model_predictions.append((boxes_mod, scores_mod, labels_mod))
        return boxes_list, scores_list, labels_list
            

    
    #runs predictions on all images in img_folder using self.ensemble to decide which method
    def predict(self, img_folder, gt_folder=None, predict_folders= []):
        boxes_list, scores_list, labels_list = [], [], []
        if img_folder is None and predict_folders is None:
            assert("No predictions to work with! Both img_folder and predict folders are empty!")
        
        img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
        img_shapes_list= []
        for img in img_paths:
            img_shapes_list.append(cv2.imread(img).shape[:2])

        # Load the image paths in the folder
        boxes_list, scores_list, labels_list = [], [], []
        #Run all input models on the input data if there are not predict folders as input.
        if img_folder is not None:
            img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
            if len(predict_folders) == 0:
                boxes_list, scores_list, labels_list =self.run_models(img_paths=img_paths)

            self.img_paths = img_paths
        #Calculate all img shapes(width, height)
        
        
        #if there are predict folders and if they are equal in ammount to the amount of models, read predictions from input files instead 
        # of doing predictions on the images.
        if len(predict_folders) != 0:
            if len(predict_folders) != len(self.models):
                assert(f"the amount of models needs to be equal to the amount of predict folders: predfolder = {len(predict_folders)} != models {len(self.models)}")
            for folder in predict_folders:
                pred_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt') or f.endswith('.xml')]
                boxesp, scoresp, labelsp = [], [], []
                for file in pred_paths:
                    boxes, scores, labels = self.read_yolo_file(file)
                    boxesp.append(boxes)
                    scoresp.append(scores)
                    labelsp.append(labels)
                self.model_predictions.append((boxesp, scoresp, labelsp))
                boxes_list.append(boxesp)
                scores_list.append(scoresp)
                labels_list.append(labelsp)

        #if there are groundtruths attatched, read them. They will be used for producing statistics later.
        if gt_folder is not None:
            gt_paths = [os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith('.txt') or f.endswith('.xml')]
            bbox_list, label_list = [], []
            for file in gt_paths:
                bboxes, labels = self.read_yolo_groundtruth_file(file)
                bbox_list.append(bboxes)
                label_list.append(labels)

            self.gts = [bbox_list, label_list]
            
                    

        
        
        
        #TODO Iterate each ensemble with iou =[0.5, 0.55. 0.6, ... 0.95] so that map50-90 can be calculated
        #Iterates over the list of ensembles
        ensembles = self.ensemble_methods
        for ensemble in ensembles:
            print(ensemble)
            #print(self.model_predictions[0])
            self.ensemble_methods = ensemble
            eboxes, escores, elabels = self.run_ensemble(img_shapes_list, boxes_list, scores_list, labels_list, img_folder)
            #print(eboxes, escores, elabels)
            ensembleResult = Ensemble(ensemble, (eboxes, escores, elabels))
            self.ensemble_results.append(ensembleResult)
        self.ensemble_methods =ensembles



    #Runs runs the result of each img on in ensemble
    def run_ensemble(self, img_shapes_list, boxes_list, scores_list, labels_list, img_folder):
        j = 0
        result_bboxes, result_scores, result_labels = [], [],[]
        #for each first img predictions from each model, should iterate once for each i images.
        for model_predictions_boxes, model_predictions_scores, model_predictions_labels in zip(zip(*boxes_list), zip(*scores_list), zip(*labels_list)):
            #(f"Doing {self.ensemble_methods} on image {j}")
            bboxes= []
            scores= []
            labels=[]
            #should iterate once for each model
            for i in range(len(model_predictions_boxes)):
                width = img_shapes_list[j][1]
                height = img_shapes_list[j][0]
                #print(model_predictions_boxes[i].shape)
                boxes= model_predictions_boxes[i]
                
                #norm_boxes = [[coord / width if idx % 2 == 0 else coord / height for idx, coord in enumerate(coords)]for coords in boxes]
                #print(len(model_predictions_boxes[i].tolist()))
                if len(boxes) != 0:
                    bboxes += [boxes]
                    scores += [model_predictions_scores[i]]
                    labels += [model_predictions_labels[i]]
                
            
            #Converting to coco which the ensemble functions require
            coco_boxes = []
            for boxes in bboxes:
                coco_boxes.append([self.yolo_to_coco_norm(box) for box in boxes])
            
            

            bboxes, scores, labels = self.pick_ensemble(coco_boxes, scores, labels)

            yolo_boxes = []
            for box in bboxes:
                yolo_boxes.append(self.coco_to_yolo_norm(box))
            
            result_bboxes.append(yolo_boxes)
            result_scores.append(scores)
            result_labels.append(labels)
            
            j+=1
        
        
        #UNCOMMENT TO SHOW IMGS
        #img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
        #self.box_imgs(model_name="YOLOV8", bboxes=result_bboxes, labels=result_labels, scores=result_scores, output_folder="test_out", img_paths=img_paths)
        return result_bboxes, result_scores, result_labels
    
    def coco_to_xyxy(self, coco_bbox):
        x_min, y_min, width, height = coco_bbox
        x_max = x_min + width
        y_max = y_min + height

        return [x_min, y_min, x_max, y_max]

    def yolo_to_xyxy(self, yolo_bbox):
        x_center, y_center, width, height = yolo_bbox
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        return [x_min, y_min, x_max, y_max]
    def yolo_to_coco_norm(self, yolo_bbox):
        x_center, y_center, width, height = yolo_bbox
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        return [x_min, y_min, x_max, y_max]
    
    def coco_to_yolo_norm(self, coco_bbox):
        x_min, y_min, x_max, y_max = coco_bbox
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        return [x_center, y_center, width, height]      
    
    #picks and runs ensemble on ONE img
    #should return the new bboxes, labels, and scores for that img TBC!!
    def pick_ensemble(self, bboxes, scores, labels):
        # Combine the model predictions using the ensemble method

        if self.ensemble_methods == 'nms':
            bboxes, scores, labels = nms(bboxes, scores, labels, iou_thr=0.6)
            combined_preds = np.column_stack((bboxes, scores, labels))

        elif self.ensemble_methods == 'soft-nms':
            
            bboxes, scores, labels = soft_nms(bboxes, scores, labels, method=2, iou_thr=self.iou)
            combined_preds = np.column_stack((bboxes, scores, labels))
        elif self.ensemble_methods == 'nmw':

            bboxes, scores, labels = non_maximum_weighted(bboxes, scores, labels, iou_thr=self.iou)
            combined_preds = np.column_stack((bboxes, scores, labels))
        elif self.ensemble_methods == 'wbf':

            bboxes, scores, labels = weighted_boxes_fusion(bboxes, scores, labels, iou_thr=self.iou)
            combined_preds = np.column_stack((bboxes, scores, labels))
        elif self.ensemble_methods == "OBB": #DOES NOT WORK ATM, NEEDS FIX
            #bboxes, scores, labels = [], [], []
            subprocess.run(['python', 'program.py'])
        return bboxes, scores, labels
        
    def box_imgs(self, model_name, bboxes, scores, labels, output_folder, img_paths, format="yolo", norm=True):
        #format can be "yolo", "coco" or "xyxy"
        import pathlib
        pathlib.Path(f"{output_folder}/{model_name}").mkdir(parents=True, exist_ok=True) 
        #os.mkdir(f"test_out/{model_name}")
        i =0
        for img, bboxes_img, scores_img, labels_img in zip(img_paths, bboxes, scores, labels):
            height, width =cv2.imread(img).shape[:2]
            img1 = cv2.imread(img)
            if img1 is None:
                print(f"Failed to read image: {img}")
                continue
            annotator = Annotator(img1)
            
            for bbox, score, label in zip(bboxes_img, scores_img, labels_img):# borde baseras på i vilket det inte göra tam
                if format == "yolo":
                    bbox = self.yolo_to_xyxy(bbox)
                elif format == "coco":
                    bbox = self.coco_to_xyxy(bbox)
                elif format == "xyxy":
                    continue
                else:
                    assert(f"Formats allowed: yolo, coco or xyxy. Format {format} not allowed!")
                x1, y1, x2, y2 = bbox
                if norm:
                    x1 *=width
                    x2 *= width
                    y1 *= height
                    y2 *= height
                    bbox = x1, y1, x2, y2
                annotator.box_label(box=bbox, label=f"{label} {score}", )
            
            cv2.imshow('image',img1)
            cv2.waitKey(3000)
            #print(img)
            
            cv2.imwrite(f"{output_folder}/{model_name}/{img.split('/')[1]}", img1)
            i+=1     

    def read_yolo_groundtruth_file(self, groundtruth_file):
        boxes, labels = [], []

        with open(groundtruth_file, 'r') as file:
            for line in file.readlines():
                data = line.strip().split(' ')
                label, x_center, y_center, width, height = int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4])

                boxes.append((x_center, y_center, width, height))
                labels.append(label)
                
        return boxes, labels

    def read_yolo_file(self, prediction_file):
        boxes, scores, labels = [], [], []
        
        with open(prediction_file, 'r') as file:
            for line in file.readlines():
                data = line.strip().split(' ')
                #print(data)
                label, x_center, y_center, width, height, score = int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5])
                
                boxes.append([x_center, y_center, width, height])
                scores.append(score)
                labels.append(label)
        
        return boxes, scores, labels

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


    def get_next_folder_name(self, base_dir, folder_prefix):
        i = 0
        while True:
            folder_name = f"{folder_prefix}{i}"
            folder_path = os.path.join(base_dir, folder_name)
            if not os.path.exists(folder_path):
                return folder_path
            i += 1

    def compare_models(self):
        plt.figure(figsize=(10, 7))
        ensembles = []
        for ensemble in self.ensemble_results:
            ensembles.append(ensemble.predictions)

        base_dir = os.getcwd()
        folder_prefix = "test_out"
        test_out_dir = self.get_next_folder_name(base_dir, folder_prefix)
        os.makedirs(test_out_dir, exist_ok=True)

        f1_data = []
        gt_boxes, gt_labels = self.gts
        for model_name, data in zip(self.model_names +self.ensemble_methods , self.model_predictions + ensembles): #+ ensembles)
            pred_boxes, pred_scores, pred_labels = data
            print(model_name)

            pred_boxes = data[0]
            pred_scores = data[1]
            pred_labels = data[2]

            f1_scores, confidence_scores = self.eval_model(self.img_paths, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)

            model_dir = os.path.join(test_out_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Plot and save the F1 scores for each IoU threshold
            for iou_idx in range(f1_scores.shape[1]):
                plt.plot(confidence_scores, f1_scores[:, iou_idx], label=f"IoU {iou_idx}")

                # Customize the plot
                plt.xlabel('Confidence Threshold')
                plt.ylabel('F1 Score')
                plt.title(f'F1 Score vs Confidence Threshold for IoU {iou_idx}')
                plt.legend()
                plt.grid()

                plt.xlim(0, 1)
                plt.ylim(0, 1)

                # Save the plot
                plot_filename = os.path.join(model_dir, f"IoU{iou_idx}.png")
                plt.savefig(plot_filename)

                # Clear the plot for the next iteration
                plt.clf()
            



    def create_coco_dict(self, img_paths, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        coco_predictions = []
        coco_ground_truth = []
        images = []

        gt_id = 0
        print(f"len of predict= {len(pred_boxes)}")
        for i, img_path in enumerate(img_paths):
            image_id = i
            filename = img_path.split('/')[-1]
            shape = cv2.imread(img_path).shape[:2]
            img_height, img_width = shape

            coco_image = {
                'id': image_id,
                'file_name': filename,
                'width': img_width,
                'height': img_height
            }

            images.append(coco_image)
                
            
            for j, box in enumerate(pred_boxes[i]):
                coco_box = self.yolo_to_coco(box, img_height=img_height, img_width=img_width)
            
                _, _, width, height = coco_box
                coco_predictions.append({
                    'image_id': image_id,
                    'category_id': int(pred_labels[i][j]),
                    'bbox': coco_box,
                    'score': pred_scores[i][j],
                    'area' : width * height
                })
            
            
            for j, box in enumerate(gt_boxes[i]):
                coco_box = box
                x, y, width, height = self.yolo_to_coco(box, img_height=img_height, img_width=img_width)
                
                area = width * height
                

                coco_ground_truth.append({
                    'id': gt_id,
                    'image_id': image_id,
                    'category_id': gt_labels[i][j],
                    'bbox': [x, y, width, height],
                    'iscrowd': 0,
                    'area': area
                })
                gt_id += 1
            
            

        categories = [
            {'id': 0, 'name': 'D00'},
            {'id': 1, 'name': 'D10'},
            {'id': 2, 'name': 'D20'},
            {'id': 3, 'name': 'D40'}
        ]
        return coco_ground_truth, images, categories, coco_predictions
   
    def eval_model(self, img_paths, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        
        coco_ground_truth, images, categories, coco_predictions = self.create_coco_dict(img_paths, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)

        gt_coco = COCO()
        gt_coco.dataset = {'annotations': coco_ground_truth, 'images': images, 'categories': categories}
        gt_coco.createIndex()

        dt_coco = gt_coco.loadRes(coco_predictions)


        #print("Ground truth annotations:", len(gt_coco.dataset['annotations']))
        #print("Examples:", gt_coco.dataset['annotations'][:5])

        #print("Predictions:", len(dt_coco.dataset['annotations']))
        #print("Examples:", dt_coco.dataset['annotations'][:5])
        coco_eval = COCOeval(gt_coco, dt_coco, iouType='bbox')
        coco_eval.params.iouThrs = np.linspace(0.1, 0.95, num=18)

        # Set new area range and max detections per image
        #new_area_ranges = [[0 ** 2, 1e5 ** 2]]  # Single area range covering all sizes
        #new_max_detections = [15, 15, 15]  # Max 15 detections per image

        #coco_eval.params.areaRng = new_area_ranges
        #coco_eval.params.maxDets = new_max_detections

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        confidence_scores = np.unique(coco_eval.eval['scores'])
        return self.calculate_f1_scores(coco_eval), confidence_scores

        
    
    def yolo_to_coco(self, yolo_bbox, img_width, img_height):
        x_center, y_center, width, height = yolo_bbox
        #print("inside coversion")
        #print(yolo_bbox)

        # Check if any of the input values are outside the expected range
        if any(value < 0 or value > 1 for value in yolo_bbox):
            print("Invalid YOLO bounding box:", yolo_bbox)
        # Denormalize the coordinates and dimensions
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Calculate the top-left corner (x_min, y_min) of the bounding box
        x_min = x_center - width / 2
        y_min = y_center - height / 2

        #print("y/xmin")
        #print(x_min, y_min)

        # Return the COCO format bounding box [x_min, y_min, width, height]
        return [x_min, y_min, width, height]
    


    def calculate_f1_scores(self, coco_eval):
        # Extract precision, recall, and scores from coco_eval
        precision = coco_eval.eval['precision']
        recall = coco_eval.eval['recall']
        scores = coco_eval.eval['scores']

        # Calculate the mean precision and recall across all categories (axis=2) and area ranges (axis=3)
        mean_precision = np.mean(precision, axis=(2, 3))  # Shape: (T, R, K)
        mean_recall = np.mean(recall, axis=(2, 3))  # Shape: (T, R, K)

        # Initialize lists to store F1 scores for each confidence level
        f1_scores = []

        # Iterate through the sorted unique confidence scores
        for confidence in np.unique(scores):
            # Find the index of the first recall value greater than or equal to the current confidence
            recall_idx = np.argmax(mean_recall >= confidence, axis=1)

            # Calculate the corresponding precision values for the chosen recall index
            precision_at_confidence = mean_precision[np.arange(mean_precision.shape[0]), recall_idx, :]

            # Calculate the F1 scores for each max detections value
            f1_at_confidence = 2 * (precision_at_confidence * confidence) / (precision_at_confidence + confidence)

            # Calculate the mean F1 score across max detections values
            mean_f1_at_confidence = np.mean(f1_at_confidence, axis=1)

            # Append the mean F1 score to the list
            f1_scores.append(mean_f1_at_confidence)

        # Convert the list of F1 scores to a NumPy array
        f1_scores = np.array(f1_scores)

        print(f1_scores.shape)
        return f1_scores # (N, T) where N is conf scores, and T is the number of IoU thresholds.
    



    def graveyard():
        iou_thresholds = coco_eval.params.iouThrs
        conf_thresholds = coco_eval.params.recThrs
        precision = coco_eval.eval['precision']
        recall = coco_eval.eval['recall']
        confidences = coco_eval.eval['scores']
        print(precision.shape)
        print(recall.shape)
        
        TP = recall * precision.shape[1]
        FN = precision.shape[1] - TP

        # Calculate recall at each confidence level
        recall_at_confidence = np.zeros((precision.shape[0], precision.shape[1], precision.shape[2], precision.shape[3], precision.shape[4]))

        for t in range(precision.shape[0]):
            for r in range(precision.shape[1]):
                for k in range(precision.shape[2]):
                    for a in range(precision.shape[3]):
                        for m in range(precision.shape[4]):
                            tp = TP[t, k, a, m]
                            fn = FN[t, k, a, m]
                            recall_at_confidence[t, r, k, a, m] = tp / (tp + fn + 1e-6)
        print(recall_at_confidence.shape)
        
        F1 = 2 * (precision * recall_at_confidence) / (precision + recall_at_confidence)
        print(F1.shape)
        f1_scores_confidences = []
        #describes the iou value where the highest f1 range for prediction recide.
        iou_index_max_conf = 0
        max_f1= 0
        for iou in range(precision.shape[0]):
            iou_f1 = []
            for i in range(precision.shape[1]):
                p = np.mean(precision[iou, i, 0, :, 0])
                r = np.mean(recall_at_confidence[iou, i, 0, :, 0])
                f1 =  (2 * p *  r)/(p+r)
                iou_f1.append(f1)
                if f1 > max_f1:
                    max_f1 = f1
                    iou_index_max_conf =iou
            f1_scores_confidences.append(iou_f1)

        # Create a list of confidence levels.
        confidence_levels = np.linspace(0, 1, 101)
        #print(f1_scores_confidences.shape)
        #print(max_f1)
        #print(iou_index_max_conf)
        #print(confidence_levels)
        #print(F1)
        return max_f1, iou_index_max_conf, confidence_levels, F1