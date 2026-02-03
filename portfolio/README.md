# Traffic Vehicle Detection using YOLOv8

## Overview
This project demonstrates a custom-trained YOLOv8 model for vehicle detection in traffic videos.
The model was trained from scratch on a custom dataset and tested on real-world traffic footage.

## Model
- Architecture: YOLOv8n
- Framework: Ultralytics YOLOv8
- Training device: CPU (Intel i5)
- Epochs: 50
- Image size: 640

## Results
- mAP50: 0.93+
- Precision: ~0.95
- Recall: ~0.86

## Demo
See `demo_video.mp4` for detection results on a traffic video.

## Notes
- Some vehicles (trucks, rear-view cars) may be missed due to dataset limitations.
- Model performance will be improved with more diverse training data.

## Tools
- Python
- OpenCV
- PyTorch
- YOLOv8
