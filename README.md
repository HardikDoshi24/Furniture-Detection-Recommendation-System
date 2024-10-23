# Furniture Detection and Recommendation System

## Overview

This project aims to enhance the user experience in **interior design** by leveraging **object detection** and **recommendation algorithms** to suggest furniture items based on their color and structural similarity. The system detects furniture items from room images and recommends visually similar furniture using **Faster R-CNN** for object detection, **VGG16** for feature extraction, and **FAISS** for similarity search.

The system is built using **Streamlit** for an interactive web interface, allowing users to upload images of rooms and receive real-time furniture recommendations that match their existing decor.

## Features

- **Object Detection**: Identifies furniture items such as chairs, couches, beds, and tables using the Faster R-CNN model.
- **Color Matching**: Recommends furniture based on dominant color extraction using RGB histograms.
- **Texture and Shape Matching**: Uses VGG16 to extract feature vectors and matches similar furniture based on structure and texture.
- **Interactive Interface**: Built using Streamlit, allowing users to upload room images and get recommendations instantly.

## Tech Stack

- **Frontend**: Streamlit
- **Object Detection**: Faster R-CNN (PyTorch)
- **Feature Extraction**: VGG16 (Keras/TensorFlow)
- **Similarity Search**: FAISS (Facebook AI Similarity Search)
- **Color Matching**: RGB Histograms and KMeans clustering
- **Backend**: Python, OpenCV, Pandas, Keras, PyTorch

## Prerequisites

- Python 3.8+
- Streamlit
- PyTorch
- TensorFlow/Keras
- OpenCV
- FAISS
- Scikit-learn


