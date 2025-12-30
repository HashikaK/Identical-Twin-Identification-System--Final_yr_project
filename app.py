import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageEnhance

app = Flask(__name__)

# -------------------- Model Setup --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print(f"✅ Using device: {device}")

# -------------------- Globals --------------------
known_faces = {}
current_detected_name = "Idle — click the button to start detection"
result_type = "idle"
detection_in_progress = False

SIMILARITY_THRESHOLD = 0.82
CONSECUTIVE_REQUIRED = 3
FRAME_WIDTH = 480
FRAME_HEIGHT = 360

# -------------------- Utility --------------------
def preprocess_image(img):
    """Enhance contrast and brightness for consistent embeddings."""
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    return img

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -------------------- Registration --------------------
def register_person_folder(name, folder_path):
    """Register a person from multiple images."""
    embeddings = []
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        path = os.path.join(folder_path, filename)
        img = Image.open(path).convert('RGB')
        img = preprocess_image(img)
        face_tensor = mtcnn(img)
        if face_tensor is None:
            continue
        with torch.no_grad():
            emb = facenet(face_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)

    if embeddings:
        mean_emb = np.mean(embeddings, axis=0)
        mean_emb /= np.linalg.norm(mean_emb)
        known_faces[name] = mean_emb
        print(f"✅ Registered {name} ({len(embeddings)} images)")
    else:
        print(f"⚠️ No valid faces found for {name}")

# Register persons
register_person_folder("Hashika", "uploads/hashika")
register_person_folder("Hayanthika", "uploads/hayanthika")

# -------------------- Detection Logic --------------------
def generate_frames():
    """Capture webcam stream, detect one face, and stop after confirmation."""
    global current_detected_name, result_type, detection_in_progress
    detection_in_progress = True
    current_detected_name = "Detecting..."
    result_type = "detecting"

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    last_candidate = None
    consecutive_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess live frame for brightness normalization
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_tensor = mtcnn(img)
            overlay_msg = "Detecting..."

            if face_tensor is not None:
                with torch.no_grad():
                    emb = facenet(face_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()
                emb = emb / np.linalg.norm(emb)

                best_name = None
                best_sim = -1.0
                second_best = -1.0

                for name, known_emb in known_faces.items():
                    sim = cosine_similarity(emb, known_emb)
                    print(f"🧠 Similarity with {name}: {sim:.3f}")
                    if sim > best_sim:
                        second_best = best_sim
                        best_sim = sim
                        best_name = name
                    elif sim > second_best:
                        second_best = sim

                diff_margin = best_sim - second_best
                threshold = SIMILARITY_THRESHOLD
                if diff_margin > 0.05 and best_sim > (SIMILARITY_THRESHOLD - 0.05):
                    threshold = SIMILARITY_THRESHOLD - 0.05

                if best_sim >= threshold:
                    candidate = best_name
                else:
                    candidate = "Unknown"

                if candidate == last_candidate:
                    consecutive_count += 1
                else:
                    last_candidate = candidate
                    consecutive_count = 1

                overlay_msg = candidate if candidate != "Unknown" else "Unknown person"

                if consecutive_count >= CONSECUTIVE_REQUIRED:
                    if candidate != "Unknown":
                        current_detected_name = f"✅ Detected: {candidate} ... Detection complete"
                        result_type = "known"
                    else:
                        current_detected_name = f"❌ Unknown Person ... Detection complete"
                        result_type = "unknown"

                    detection_in_progress = False
                    color = (0, 255, 0) if result_type == "known" else (0, 0, 255)
                    cv2.putText(frame, overlay_msg, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    break

            cv2.putText(frame, overlay_msg, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        cap.release()
        detection_in_progress = False
        if result_type == "detecting":
            current_detected_name = "❌ No face detected ... Detection complete"
            result_type = "unknown"

# -------------------- Routes --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global current_detected_name, result_type, detection_in_progress
    current_detected_name = "Detecting..."
    result_type = "detecting"
    detection_in_progress = True
    _ = request.args.get('ts', None)
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_name')
def get_name():
    return jsonify({
        "name": current_detected_name,
        "result": result_type,
        "in_progress": detection_in_progress
    })

if __name__ == '__main__':
    app.run(debug=True)
