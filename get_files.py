import os
import shutil

predictions_folder = 'test_pred_folder'
annotations_folder = '/home/martin/Desktop/RDD_proj/datasets/RDD2022/RDD'
images_folder = '/home/martin/Desktop/RDD_proj/datasets/RDD2022/RDD'

test_input_folder = 'test_input'
test_gt_folder = 'test_gt'

# Create the test_input and test_gt folders if they don't exist
os.makedirs(test_input_folder, exist_ok=True)
os.makedirs(test_gt_folder, exist_ok=True)

# Get a list of prediction files
prediction_files = [f for f in os.listdir(predictions_folder) if f.endswith('.txt')]

# Iterate through each prediction file and copy the corresponding annotation and image files
for pred_file in prediction_files:
    # Remove .txt extension from the prediction file name
    file_name_without_ext = os.path.splitext(pred_file)[0]

    # Check if the corresponding annotation file exists
    annotation_file = file_name_without_ext + '.txt'
    annotation_file_path = os.path.join(annotations_folder, annotation_file)
    if os.path.exists(annotation_file_path):
        # Copy the annotation file to the test_gt folder
        shutil.copy(annotation_file_path, os.path.join(test_gt_folder, annotation_file))

    # Check if the corresponding image file exists
    image_file = file_name_without_ext + '.jpg'  # Change to the appropriate image extension (e.g., .png, .jpeg, etc.)
    image_file_path = os.path.join(images_folder, image_file)
    if os.path.exists(image_file_path):
        # Copy the image file to the test_input folder
        shutil.copy(image_file_path, os.path.join(test_input_folder, image_file))
