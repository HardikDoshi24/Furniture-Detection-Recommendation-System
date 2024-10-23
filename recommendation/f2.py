import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
import shutil
import pandas as pd
import pickle
import shutil
import faiss
from PIL import Image, ImageOps
from keras.models import load_model
from keras.preprocessing import image as keras_image
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn import preprocessing

# ----------------------------
# 1. Setup and Model Initialization
# ----------------------------

st.set_page_config(page_title="Furniture Detection & Recommendation", layout="wide")

# Initialize object detection model (Faster R-CNN)
@st.cache_resource
def load_detection_model():
    model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model_.eval()
    return model_

detection_model = load_detection_model()

# Define COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
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

# Initialize recommendation model (VGG16-based)
@st.cache_resource
def load_recommendation_model():
    weights_file = 'sdp2/vgg16_furniture_classifier_1129.keras'
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.load_weights(weights_file)
    
    # Load model_2 for recommendations (assuming model_2 is the same as model)
    model_2 = load_model(weights_file)
    
    return model, model_2

recommendation_model, recommendation_model_2 = load_recommendation_model()

# ----------------------------
# 2. Utility Functions
# ----------------------------

# Object Detection Functions
def get_predictions(pred, threshold=0.8, objects=None):
    """
    Process the raw predictions from the detection model and filter them based on confidence threshold and desired object classes.

    Args:
        pred (dict): Raw predictions from the model.
        threshold (float): Confidence score threshold.
        objects (list): List of object names to filter.

    Returns:
        list: List of tuples containing (object_name, confidence_score, bounding_box).
    """
    # Extract relevant data from predictions
    predicted_classes = [(COCO_INSTANCE_CATEGORY_NAMES[i], p, box) 
                         for i, p, box in zip(pred[0]['labels'].numpy(), 
                         pred[0]['scores'].detach().numpy(), pred[0]['boxes'].detach().numpy())]
    
    # Filter out predictions below the confidence threshold
    predicted_classes = [item for item in predicted_classes if item[1] > threshold]
    
    # If specific objects are provided, filter accordingly
    if objects and predicted_classes:
        predicted_classes = [(name, p, box) for name, p, box in predicted_classes if name in objects]
    return predicted_classes

def draw_boxes(pred_classes, img, rect_th=2, text_size=0.5, text_th=2):
    """
    Draw bounding boxes and labels on the image.

    Args:
        pred_classes (list): List of detected classes with their confidence scores and bounding boxes.
        img (PIL.Image): The original image.
        rect_th (int): Thickness of the rectangle border.
        text_size (float): Size of the text.
        text_th (int): Thickness of the text.

    Returns:
        numpy.ndarray: Image with drawn bounding boxes and labels.
    """
    image = np.array(img).copy()

    for cls in pred_classes:
        label, prob, box = cls
        t, l, r, b = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        cv2.rectangle(image, (t, l), (r, b), (0, 255, 0), rect_th)
        cv2.putText(image, f"{label}: {prob:.2f}", (t, l-10), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), text_th)

    return image

def crop_detected_objects(pred_classes, img, save_folder="sdp2/uploaded_images/cropped"):
    """
    Crop detected objects from the image and save them locally.

    Args:
        pred_classes (list): List of detected classes with their bounding boxes.
        img (PIL.Image): The original image.
        save_folder (str): Directory to save cropped images.

    Returns:
        list: List of tuples containing (object_label, cropped_image_path).
    """
    cropped_images = []
    image_np = np.array(img).copy()

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for idx, cls in enumerate(pred_classes):
        label, _, box = cls
        t, l, r, b = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Ensure coordinates are within image bounds
        h, w, _ = image_np.shape
        t = max(0, min(t, w - 1))
        l = max(0, min(l, h - 1))
        r = max(0, min(r, w - 1))
        b = max(0, min(b, h - 1))

        cropped_img = image_np[l:b, t:r]
        if cropped_img.size == 0:
            st.warning(f"Cropped image for {label} is empty. Skipping.")
            continue
        cropped_image_pil = Image.fromarray(cropped_img)

        # Generate unique filename to avoid conflicts
        cropped_image_path = os.path.join(save_folder, f"cropped_{label}_{idx}.jpg")
        cropped_image_pil.save(cropped_image_path)

        cropped_images.append((label, cropped_image_path))

    return cropped_images

# Recommendation Functions

# Removed `predict_class` function since we will use Faster R-CNN's prediction directly

def get_dominant_color(image, k=1):
    """
    Extract the dominant color from the image using K-Means clustering.

    Args:
        image (numpy.ndarray): The image in BGR format.
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: Dominant color in RGB.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_[0]

def color_distance(color1, color2):
    """
    Calculate the Euclidean distance between two colors.

    Args:
        color1 (numpy.ndarray): First color (RGB).
        color2 (numpy.ndarray): Second color (RGB).

    Returns:
        float: Euclidean distance between the two colors.
    """
    return np.linalg.norm(color1 - color2)

def find_all_matches(uploaded_color, csv_folder, class_name, threshold=20):
    """
    Find all images that match the uploaded color within a threshold.

    Args:
        uploaded_color (numpy.ndarray): Dominant color of the uploaded image.
        csv_folder (str): Directory containing CSV files with color information.
        class_name (str): Class name of the uploaded image.
        threshold (float): Color distance threshold.

    Returns:
        list: List of matching image names.
    """
    csv_path = os.path.join(csv_folder, f"{class_name}_colors.csv")
    if not os.path.exists(csv_path):
        st.warning(f"Color CSV for class '{class_name}' not found.")
        return []
    df = pd.read_csv(csv_path)

    matching_images = []
    for _, row in df.iterrows():
        dataset_color = np.array([row['R'], row['G'], row['B']])
        dist = color_distance(uploaded_color, dataset_color)
        if dist <= threshold:
            matching_images.append(row['Image_Name'])

    return matching_images

def copy_images_to_folder(image_names, source_folder, destination_folder):
    """
    Copy matched images from source to destination folder.

    Args:
        image_names (list): List of image file names to copy.
        source_folder (str): Source directory.
        destination_folder (str): Destination directory.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for image_name in image_names:
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
        else:
            st.error(f"Image {image_name} not found in source folder {source_folder}.")

def process_uploaded_image(class_name, csv_folder, image_folder, threshold=20):
    """
    Process the detected class to find matches based on color.

    Args:
        class_name (str): Detected class name.
        csv_folder (str): Directory containing CSV files with color information.
        image_folder (str): Directory containing all images categorized by class.
        threshold (float): Color distance threshold.

    Returns:
        tuple: Detected class name and list of matching image names.
    """
    # Assume the image has already been read and color extracted outside this function
    # This function now only finds matches based on the class name and color
    # You need to pass the dominant color as an argument or handle it accordingly

    # For this integration, we'll handle color extraction outside and pass it as needed
    pass  # Placeholder since color extraction is handled elsewhere

def load_matched_features(matched_folder, predicted_class):
    """
    Load precomputed feature vectors and file mappings for matched images.

    Args:
        matched_folder (str): Directory containing matched images.
        predicted_class (str): Predicted class name.

    Returns:
        tuple: Array of matched feature vectors and list of their file paths.
    """
    class_path = f"sdp2/file_mapping/{predicted_class}"
    features_path = os.path.join(class_path, f"V_1_norm_{predicted_class}.p")
    file_mapping_path = os.path.join(class_path, f"file_mapping_{predicted_class}.p")

    if not os.path.exists(features_path) or not os.path.exists(file_mapping_path):
        st.error(f"Feature or mapping file missing for class '{predicted_class}'.")
        return np.array([]), []

    with open(features_path, 'rb') as f:
        V = pickle.load(f)
    with open(file_mapping_path, 'rb') as f:
        file_mapping = pickle.load(f)

    matched_filenames = os.listdir(matched_folder)
    matched_features = []
    matched_file_mapping = []

    for idx, file_path in file_mapping.items():
        file_name = os.path.basename(file_path)
        if file_name in matched_filenames:
            matched_features.append(V[idx])
            relative_path = os.path.join("sdp2/images", predicted_class, file_name)
            matched_file_mapping.append(relative_path)

    return np.array(matched_features), matched_file_mapping

def extract_feature_vector(img_path, model, fc_extractor, target_size):
    """
    Extract and normalize the feature vector from the image using a specific model layer.

    Args:
        img_path (str): Path to the image.
        model (tf.keras.Model): The trained model.
        fc_extractor (tf.keras.Model): Model to extract features from a specific layer.
        target_size (tuple): Target size for image preprocessing.

    Returns:
        numpy.ndarray: Normalized feature vector.
    """
    img_data, _ = image_upload(img_path, target_size)
    feature_vector = fc_extractor.predict(img_data)
    return preprocessing.normalize(feature_vector, norm="l2").reshape(-1,)

def similarity_search(V, v_query, file_mapping, n_results=6):
    """
    Perform similarity search using FAISS to find the most similar images.

    Args:
        V (numpy.ndarray): Array of feature vectors.
        v_query (numpy.ndarray): Feature vector of the query image.
        file_mapping (list): List mapping indices to file paths.
        n_results (int): Number of similar images to retrieve.

    Returns:
        list: List of file paths for similar images.
    """
    if V.size == 0:
        return []
    v_query = np.expand_dims(v_query, axis=0)
    d = V.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(V))
    distances, closest_indices = index.search(v_query, n_results)
    distances = distances.flatten()
    closest_indices = closest_indices.flatten()

    closest_paths = [file_mapping[idx] for idx in closest_indices if idx < len(file_mapping)]
    return closest_paths

def image_upload(img_path, target_size):
    """
    Upload and preprocess the image.

    Args:
        img_path (str): Path to the image.
        target_size (tuple): Desired image size.

    Returns:
        tuple: Preprocessed image array and PIL image object.
    """
    img = ImageOps.fit(Image.open(img_path), target_size, Image.Resampling.LANCZOS)
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x, img

def display_images(image_paths, captions=None, width=150):
    """
    Display images in Streamlit.

    Args:
        image_paths (list): List of image file paths.
        captions (list, optional): List of captions for the images.
        width (int, optional): Width of the displayed images.
    """
    if not image_paths:
        st.write("No images to display.")
        return
    images = [Image.open(img_path) for img_path in image_paths]
    if captions:
        st.image(images, caption=captions, width=width)
    else:
        st.image(images, width=width)

def clear_folder(classname):
    folder_path = os.path.join('sdp2', 'matched_images', f"{classname}_matches")
    
    if os.path.exists(folder_path):
        # Delete all files inside the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Remove file
                    
            except Exception as e:
                st.error(f"Error deleting file {file_path}: {e}")
        st.success(f"All files in {folder_path} have been deleted.")
    else:
        st.warning(f"Folder {folder_path} does not exist.")

# ----------------------------
# 3. Streamlit Interface
# ----------------------------

def main():
    st.title("🛋️ Furniture Detection & Recommendation System")
    st.write("Upload a room image, detect furniture items, and get personalized recommendations.")

    # Sidebar Configuration (Optional)
    st.sidebar.header("Settings")
    detection_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.5, 0.99, 0.8, 0.01)
    color_threshold = st.sidebar.slider("Color Distance Threshold", 10, 100, 20, 5)

    # Image Uploader
    uploaded_file = st.file_uploader("Upload an image of a room", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Save uploaded image
        uploaded_image_path = os.path.join("sdp2/uploaded_images", uploaded_file.name)
        with open(uploaded_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Object Detection
        st.header("🔍 Detected Furniture Items")
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image)

        with torch.no_grad():
            pred = detection_model(img_tensor.unsqueeze(0))
        
        # Define objects of interest
        objects_of_interest = ["bed", "dining table", "chair", "couch"]  # Update as needed
        pred_classes = get_predictions(pred, threshold=detection_threshold, objects=objects_of_interest)

        if pred_classes:
            st.success(f"Detected {len(pred_classes)} furniture item(s).")
            st.write("### Detected Items:")
            for cls in pred_classes:
                st.write(f"- **{cls[0]}** with confidence {cls[1]:.2f}")
            
            # Draw bounding boxes
            img_with_boxes = draw_boxes(pred_classes, image)
            st.image(img_with_boxes, caption="Image with Detected Furniture", use_column_width=True)

            # Crop detected objects
            st.write("### Cropped Detected Furniture:")
            cropped_furniture = crop_detected_objects(pred_classes, image)

            if not cropped_furniture:
                st.warning("No valid cropped images were created.")
                return

            # Display cropped images with recommendation buttons
            for label, cropped_path in cropped_furniture:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(cropped_path, caption=f"Cropped: {label}", width=200)
                with col2:
                    # Unique key for each button to prevent Streamlit warnings
                    if st.button(f"Get Recommendations for {label}", key=cropped_path):
                        # Process recommendation for this cropped image
                        st.write(f"### Recommendations for {label}")

                        # Paths and directories for recommendation
                        csv_folder = "sdp2/images/image_csv"
                        image_folder = "sdp2/images"
                        # data_dir is no longer needed since we're using detected class directly

                        # Read the cropped image to extract dominant color
                        uploaded_image = cv2.imread(cropped_path)
                        if uploaded_image is None:
                            st.error(f"Failed to read image from {cropped_path}.")
                            continue
                        uploaded_color = get_dominant_color(uploaded_image)

                        # Find matching images based on color
                        matches = find_all_matches(uploaded_color, csv_folder, label, threshold=color_threshold)

                        if not matches:
                            st.info("No recommendations found based on color similarity.")
                            continue

                        # Copy matched images to a temporary folder (if needed)
                        destination_folder = f"sdp2/matched_images/{label}_matches"
                        class_image_folder = os.path.join(image_folder, label)
                        copy_images_to_folder(matches, class_image_folder, destination_folder)

                        # Load matched features
                        matched_folder = f'sdp2/matched_images/{label}_matches'
                        matched_features, matched_file_mapping = load_matched_features(matched_folder, label)

                        if matched_features.size == 0:
                            st.info("No matched features available for similarity search.")
                            continue

                        # Define feature extractor (fc1 layer)
                        fc1_extractor = tf.keras.Model(inputs=recommendation_model.input, outputs=recommendation_model.get_layer('fc1').output)
                        # Extract feature vector from the cropped image
                        compare_vector = extract_feature_vector(cropped_path, recommendation_model_2, fc1_extractor, recommendation_model.input_shape[1:3])

                        # Perform similarity search
                        similar_images = similarity_search(matched_features, compare_vector, matched_file_mapping, n_results=5)

                        if similar_images:
                            st.write("#### Similar Items:")
                            captions = [os.path.basename(p) for p in similar_images]
                            display_images(similar_images, captions=captions, width=150)
                            clear_folder(label)
                        else:
                            st.info("No similar items found.")
        else:
            st.warning("No furniture items detected with the specified confidence threshold.")

if __name__ == "__main__":
    main()
