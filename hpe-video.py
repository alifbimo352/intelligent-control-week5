from ultralytics import YOLO

#load model yolov8 pose
model = YOLO ("yolov8n-pose.pt")

#deteksi pose pada video
result = model ("video_input.mp4", save=True, show=True)