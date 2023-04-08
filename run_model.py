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
                    boxes_mod.append(raw_preds.pred[i][:, :4].cpu().numpy()) # x1, y1, x2, y2
                    scores_mod.append(raw_preds.pred[i][:, 4].cpu().numpy())
                    labels_mod.append(raw_preds.pred[i][:, 5].cpu().numpy())
            if modelv == "yolov8":
                for result in raw_preds:
                    result = result.boxes
                    
                    boxes_mod.append(result.xywhn.cpu().numpy()) # x1, y1, x2, y2
                    scores_mod.append(result.conf.cpu().numpy())
                    labels_mod.append(result.cls.cpu().numpy())

            boxes_list.append(boxes_mod)
            scores_list.append(scores_mod)
            labels_list.append(labels_mod)
            self.model_predictions.append((boxes_mod, scores_mod, labels_mod))
            

    
    #runs predictions on all images in img_folder using self.ensemble to decide which method
    def predict(self, img_folder=None, gt_folder=None, predict_folders= []):
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
                self.run_models(img_paths=img_paths)

                #img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
                #boxes_list, scores_list, labels_list  = self.model_predictions[0]
                
                #self.box_imgs(model_name="YOLOV8asdsdasda", bboxes=boxes_list, labels=labels_list, scores=scores_list, output_folder="test_out", img_paths=img_paths)

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

        #if there are groundtruths attatched, read them. They will be used for producing statistics later.
        if gt_folder is not None:
            gt_paths = [os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith('.txt') or f.endswith('.xml')]
            bbox_list, label_list = [], []
            for file in gt_paths:
                #ALREADY IN COCO FORMAT HERE
                bboxes, labels = self.read_yolo_groundtruth_file(file)
                bbox_list.append(bboxes)
                label_list.append(labels)

            self.gts = [bbox_list, label_list]
            
                    

        
        
        
        #TODO Iterate each ensemble with iou =[0.5, 0.55. 0.6, ... 0.95] so that map50-90 can be calculated
        #Iterates over the list of ensembles
        ensembles = self.ensemble_methods
        for ensemble in ensembles:
            self.ensemble_methods = ensemble
            eboxes, escores, elabels = self.run_ensemble(img_shapes_list, boxes_list, scores_list, labels_list, img_folder)
            ensembleResult = Ensemble(ensemble, (eboxes, escores, elabels))
            self.ensemble_results.append(ensembleResult)
        self.ensemble_methods =ensembles



    #Runs runs the result of each img on in ensemble
    def run_ensemble(self, img_shapes_list, boxes_list, scores_list, labels_list, img_folder):
        j = 0
        result_bboxes, result_scores, result_labels = [], [],[]
        #for each first img predictions from each model, should iterate once for each i images.
        for model_predictions_boxes, model_predictions_scores, model_predictions_labels in zip(zip(*boxes_list), zip(*scores_list), zip(*labels_list)):
            print(f"Doing {self.ensemble_methods} on image {j}")
            bboxes= []
            scores= []
            labels=[]
            #should iterate once for each model
            for i in range(len(model_predictions_boxes)):
                width = img_shapes_list[j][1]
                height = img_shapes_list[j][0]

                boxes= model_predictions_boxes[i].tolist()

                norm_boxes = [[coord / width if idx % 2 == 0 else coord / height for idx, coord in enumerate(coords)]for coords in boxes]

                bboxes += [norm_boxes]
                scores += [model_predictions_scores[i].tolist()]
                labels += [model_predictions_labels[i].tolist()]

            j+=1
            bboxes, scores, labels = self.pick_ensemble(bboxes, scores, labels)
            
            
            result_bboxes.append(bboxes)
            result_labels.append(labels)
            result_scores.append(scores)
        
        result_bboxes = self.denormalize_bboxes_array(result_bboxes, img_shapes_list)
        #print(result_bboxes)
        #UNCOMMENT TO SHOW IMGS
        #img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
        #self.box_imgs(model_name="YOLOV8", bboxes=result_bboxes, labels=result_labels, scores=result_scores, output_folder="test_out", img_paths=img_paths)
        return result_bboxes, result_scores, result_scores

    def denormalize_bboxes_array(self, bboxes_array, img_shapes):
        denormalized_bboxes_array = []
        for bboxes, (original_height, original_width) in zip(bboxes_array, img_shapes):
            denormalized_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                x1 = x1 * original_width
                y1 = y1 * original_height
                x2 = x2 * original_width
                y2 = y2 * original_height
                denormalized_bbox = [x1, y1, x2, y2]
                denormalized_bboxes.append(denormalized_bbox)
            denormalized_bboxes_array.append(np.array(denormalized_bboxes))
        return denormalized_bboxes_array        
    
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
        

    def box_imgs(self, model_name, bboxes, scores, labels, output_folder, img_paths):
        import pathlib
        pathlib.Path(f"{output_folder}/{model_name}").mkdir(parents=True, exist_ok=True) 
        #os.mkdir(f"test_out/{model_name}")
        i =0
        for img, bboxes_img, scores_img, labels_img in zip(img_paths, bboxes, scores, labels):
            img1 = cv2.imread(img)
            if img1 is None:
                print(f"Failed to read image: {img}")
                continue
            annotator = Annotator(img1)
            for bbox, score, label in zip(bboxes_img, scores_img, labels_img):# borde baseras på i vilket det inte göra tam

                #print(score)
                #print(label)
                #print(bbox)
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
                label, x_center, y_center, width, height, score = int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5])
                
                # Convert x_center, y_center, width, height to xmin, ymin, xmax, ymax
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2
                
                boxes.append([xmin, ymin, xmax, ymax])
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

    def compare_models(self):
        plt.figure(figsize=(10, 7))
        ensembles = []
        for ensemble in self.ensemble_results:
            ensembles.append(ensemble.predictions)
        
        #print(len(self.model_predictions[0]))
        #print(ensembles)
        
        conf_thresholds =np.linspace(0, 1, 101)
        gt_boxes, gt_labels = self.gts
        print(self.gts)
        for model_name, data in zip(self.model_names, self.model_predictions): #+ ensembles)
            #print(self.model_predictions[0][2])
            pred_boxes, pred_scores, pred_labels = data
            print(model_name)
            #print(data[2])
            pred_boxes = data[0]
            pred_scores = data[1]
            pred_labels = data[2]
            print("OREDPSDASD")
            print(pred_boxes)
            
            

            
            self.eval_model(self.img_paths, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
            

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left')
        plt.grid()
        plt.show()
        plt.savefig('test.png')
   
    def eval_model(self, img_paths, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        coco_predictions = []
        coco_ground_truth = []
        images = []

        gt_id = 0

        for i, img_path in enumerate(img_paths):
            image_id = i
            filename = img_path.split('/')[-1]
            print(filename)
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
                #coco_box = self.convert_to_coco_format(coco_box)
                x, y, width, height = self.yolo_to_coco(box, img_height=img_height, img_width=img_width)
                
                

                area = width * height
                
                print("GT")
                #print([x, y, width, height])
                print(coco_box)
                print(gt_labels[i][j])
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

        gt_coco = COCO()
        gt_coco.dataset = {'annotations': coco_ground_truth, 'images': images, 'categories': categories}
        gt_coco.createIndex()

        dt_coco = gt_coco.loadRes(coco_predictions)


        print("Ground truth annotations:", len(gt_coco.dataset['annotations']))
        print("Examples:", gt_coco.dataset['annotations'][:5])

        print("Predictions:", len(dt_coco.dataset['annotations']))
        print("Examples:", dt_coco.dataset['annotations'][:5])
        coco_eval = COCOeval(gt_coco, dt_coco, iouType='bbox')

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        precision = coco_eval.stats[0]
        recall = coco_eval.stats[1]
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    def yolo_to_coco(self, yolo_bbox, img_width, img_height):
        x_center, y_center, width, height = yolo_bbox

        # Denormalize the coordinates and dimensions
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Calculate the top-left corner (x_min, y_min) of the bounding box
        x_min = x_center - width / 2
        y_min = y_center - height / 2

        # Return the COCO format bounding box [x_min, y_min, width, height]
        return [x_min, y_min, width, height]
    