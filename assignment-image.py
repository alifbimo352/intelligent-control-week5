from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp

# Load model YOLOv8 untuk mendeteksi pose dan objek manusia
pose_model = YOLO("yolov8n-pose.pt")
object_model = YOLO("yolov8n.pt")

# Inisialisasi MediaPipe untuk deteksi tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=4,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Warna dan ketebalan garis
WARNA = (0, 255, 0)
KETEBALAN = 2

# Nama titik berdasarkan format COCO
NAMA_TITIK = [
    "HIDUNG", "MATA KIRI", "MATA KANAN", "TELINGA KIRI", "TELINGA KANAN",
    "BAHU KIRI", "BAHU KANAN", "SIKU KIRI", "SIKU KANAN",
    "PERGELANGAN KIRI", "PERGELANGAN KANAN", "PINGGUL KIRI", "PINGGUL KANAN",
    "LUTUT KIRI", "LUTUT KANAN", "PERGELANGAN KAKI KIRI", "PERGELANGAN KAKI KANAN"
]

# Fungsi untuk deteksi pose

def deteksi_pose(frame):
    results = pose_model(frame)
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        if keypoints is not None:
            for titik_manusia in keypoints:
                for i, (x, y) in enumerate(titik_manusia):
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                        cv2.putText(frame, f'{NAMA_TITIK[i]}', (int(x) + 10, int(y) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

# Fungsi untuk deteksi objek manusia

def deteksi_objek(frame):
    results = object_model(frame)
    for result in results:
        kotak = result.boxes.xyxy.cpu().numpy()
        id_kelas = result.boxes.cls.cpu().numpy()
        kepercayaan = result.boxes.conf.cpu().numpy()

        for box, class_id, conf in zip(kotak, id_kelas, kepercayaan):
            if int(class_id) == 0:  # Manusia
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(frame, f'MANUSIA {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

# Fungsi untuk deteksi tangan

def deteksi_tangan(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil_tangan = hands.process(frame_rgb)
    if hasil_tangan.multi_hand_landmarks:
        for landmark_tangan in hasil_tangan.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmark_tangan, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                                   mp_draw.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=3))

# Baca gambar input dan simpan hasilnya
input_image_path = "inputt.jpg"
output_image_path = "output.jpg"
frame = cv2.imread(input_image_path)

deteksi_objek(frame)
deteksi_pose(frame)
deteksi_tangan(frame)

cv2.imwrite(output_image_path, frame)
cv2.imshow("Deteksi Pose, Objek, dan Tangan", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Gambar telah diproses dan disimpan sebagai {output_image_path}")
