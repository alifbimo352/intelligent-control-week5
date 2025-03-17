from ultralytics import YOLO

#load model yolov8 pose 
model = YOLO("yolov8n-pose.pt")

#deteksi pose pad agambar
result = model("cipa.jpg", show=True)

#simpan hasil
result[0].save("sipazizah.jpg")