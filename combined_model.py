from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torchvision
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load YOLOv8 segmentation model
try:
    yolo_model = YOLO('yolo11n.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")

# Load pre-trained Faster R-CNN
try:
    rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    rcnn_model.eval()
except Exception as e:
    print(f"Error loading RCNN model: {e}")

# COCO class lists for YOLO and RCNN
coco_classes_rcnn = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

coco_classes_yolo = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Font for labeling
try:
    font = ImageFont.truetype("arial.ttf", 15)
except IOError:
    print("Font file not found, using default font.")
    font = ImageFont.load_default()

def compute_confusion_matrix(yolo_labels, rcnn_labels, class_list):
    # Convert labels to class names
    yolo_class_names = [class_list[label] for label in yolo_labels]
    rcnn_class_names = [class_list[label] for label in rcnn_labels]

    # Combine all classes for a complete list (optional, depends on scenario)
    all_detected_classes = sorted(set(yolo_class_names + rcnn_class_names))

    # Initialize counts for confusion matrix
    yolo_class_indices = [all_detected_classes.index(cls) for cls in yolo_class_names]
    rcnn_class_indices = [all_detected_classes.index(cls) for cls in rcnn_class_names]

    # Create confusion matrix (we will need to handle different lengths, so this is a simplified match matrix)
    matrix = np.zeros((len(all_detected_classes), len(all_detected_classes)), dtype=int)

    # Populate confusion matrix
    for yolo_cls in yolo_class_indices:
        if yolo_cls in rcnn_class_indices:
            matrix[yolo_cls, yolo_cls] += 1  # True positive (both detected same class)
        else:
            matrix[yolo_cls, :] += 1  # YOLO detected, but RCNN didn't (false positive for YOLO)

    for rcnn_cls in rcnn_class_indices:
        if rcnn_cls not in yolo_class_indices:
            matrix[:, rcnn_cls] += 1  # RCNN detected, but YOLO didn't (false positive for RCNN)

    # Plot Confusion Matrix
    display_labels = [all_detected_classes[i] for i in range(len(all_detected_classes))]

    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical')
    plt.title("YOLO vs RCNN Detection Confusion Matrix")
    plt.show()

def draw_boxes(draw, boxes, labels, scores, class_list, color="blue", source="RCNN"):
    """ Helper to draw boxes from RCNN or YOLO results. """
    for box, label, score in zip(boxes, labels, scores):
        if score < 0.7:
            continue
        class_name = class_list[label] if label < len(class_list) else f"Unknown {label}"
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{source}: {class_name} {score:.2f}"
        text_size = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([x1, y1 - (text_size[3] - text_size[1]), x1 + (text_size[2] - text_size[0]), y1], fill=color)
        draw.text((x1, y1 - (text_size[3] - text_size[1])), text, fill="white", font=font)


def contextual_analysis(objects, draw):
    """ Analyze the context of detected objects to provide deeper insights. """
    if len(objects) > 1:
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i + 1:]:
                if obj1['label'] == 'person' and obj2['label'] == 'car':
                    if abs(obj1['box'][0] - obj2['box'][0]) < 50:  # More precise proximity measure
                        draw.text((obj1['box'][0], obj1['box'][1] - 20), "Close to car", fill="red", font=font)


def process_and_annotate(image_path, output_path='combined_output.png'):
    # Open the image and create a drawing context
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    # Run YOLO
    yolo_results = yolo_model(image_path)
    yolo_boxes = []
    yolo_labels = []
    yolo_scores = []
    for result in yolo_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        yolo_boxes.extend(boxes)
        yolo_labels.extend(labels)
        yolo_scores.extend(scores)
        draw_boxes(draw, boxes, labels, scores, coco_classes_yolo, color="red", source="YOLO")

    # Run Faster R-CNN
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        rcnn_result = rcnn_model(image_tensor)[0]
    rcnn_boxes = rcnn_result['boxes'].cpu().numpy()
    rcnn_labels = rcnn_result['labels'].cpu().numpy()
    rcnn_scores = rcnn_result['scores'].cpu().numpy()
    draw_boxes(draw, rcnn_boxes, rcnn_labels, rcnn_scores, coco_classes_rcnn, color="blue", source="RCNN")

    # Prepare data for contextual analysis from both YOLO and RCNN results
    combined_objects = [
        {'box': box, 'label': coco_classes_rcnn[label], 'score': score}
        for box, label, score in zip(rcnn_boxes, rcnn_labels, rcnn_scores) if score >= 0.7
    ] + [
        {'box': box, 'label': coco_classes_yolo[label], 'score': score}
        for box, label, score in zip(yolo_boxes, yolo_labels, yolo_scores) if score >= 0.7
    ]

    # Call contextual analysis
    contextual_analysis(combined_objects, draw)

    # Save the image after all annotations
    image.save(output_path)
    print(f"Saved combined output to {output_path}")

    # Generate a bar graph for both YOLO and RCNN model accuracy
    plot_comparative_accuracy(yolo_labels, yolo_scores, rcnn_labels, rcnn_scores)

    compute_confusion_matrix(yolo_labels, rcnn_labels, coco_classes_rcnn)


def plot_comparative_accuracy(yolo_labels, yolo_scores, rcnn_labels, rcnn_scores):
    # Convert lists to numpy arrays for advanced indexing
    yolo_labels = np.array(yolo_labels)
    yolo_scores = np.array(yolo_scores)
    rcnn_labels = np.array(rcnn_labels)
    rcnn_scores = np.array(rcnn_scores)

    # Unique labels across both models
    all_labels = np.unique(np.concatenate([yolo_labels, rcnn_labels]))

    # Calculate average scores for YOLO
    yolo_avg_scores = [np.mean(yolo_scores[yolo_labels == label]) if label in yolo_labels else 0 for label in all_labels]

    # Calculate average scores for RCNN
    rcnn_avg_scores = [np.mean(rcnn_scores[rcnn_labels == label]) if label in rcnn_labels else 0 for label in all_labels]

    # Bar width
    bar_width = 0.35

    # X locations for the groups
    r1 = np.arange(len(all_labels))
    r2 = [x + bar_width for x in r1]

    # Create bars
    plt.bar(r1, yolo_avg_scores, color='red', width=bar_width, edgecolor='grey', label='YOLO')
    plt.bar(r2, rcnn_avg_scores, color='blue', width=bar_width, edgecolor='grey', label='RCNN')

    # Add xticks on the middle of the group bars
    plt.xlabel('Object Category', fontweight='bold')
    plt.ylabel('Average Confidence Score')
    plt.title('Comparison of YOLO and RCNN Model Accuracies')
    plt.xticks([r + bar_width/2 for r in range(len(all_labels))], [coco_classes_rcnn[label] for label in all_labels], rotation=90)

    # Create legend & Show graphic
    plt.legend()
    plt.show()


if __name__ == '__main__':
    image_path = '../image1.jpeg'  # Update with your image path
    process_and_annotate(image_path, 'combined_output.png')
