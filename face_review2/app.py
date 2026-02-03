import os
import torch
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# =======================
# Flask setup
# =======================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =======================
# Model setup
# =======================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    image_size=160,
    margin=40,
    keep_all=True,
    post_process=True,
    device=device
)

facenet = InceptionResnetV1(
    pretrained='vggface2'
).eval().to(device)

print(f"Using device: {device}")

# =======================
# Feature extraction (ROBUST)
# =======================
def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    faces = mtcnn(img)

    if faces is None:
        return None

    embeddings = []

    for face in faces:
        face = face.to(device)

        # ----- Original -----
        with torch.no_grad():
            emb = facenet(face.unsqueeze(0))
        emb = emb.squeeze().cpu().numpy()
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)

        # ----- Horizontal Flip -----
        h_flip = torch.flip(face, dims=[2])
        with torch.no_grad():
            emb_h = facenet(h_flip.unsqueeze(0))
        emb_h = emb_h.squeeze().cpu().numpy()
        emb_h = emb_h / np.linalg.norm(emb_h)
        embeddings.append(emb_h)

        # ----- Vertical Flip -----
        v_flip = torch.flip(face, dims=[1])
        with torch.no_grad():
            emb_v = facenet(v_flip.unsqueeze(0))
        emb_v = emb_v.squeeze().cpu().numpy()
        emb_v = emb_v / np.linalg.norm(emb_v)
        embeddings.append(emb_v)

        # ----- Horizontal + Vertical Flip -----
        hv_flip = torch.flip(face, dims=[1, 2])
        with torch.no_grad():
            emb_hv = facenet(hv_flip.unsqueeze(0))
        emb_hv = emb_hv.squeeze().cpu().numpy()
        emb_hv = emb_hv / np.linalg.norm(emb_hv)
        embeddings.append(emb_hv)

    # Average embedding
    return np.mean(embeddings, axis=0)

# =======================
# Distance functions
# =======================
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_similarity(a, b):
    return np.dot(a, b)

# =======================
# Flask route
# =======================
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        img1 = request.files.get('image1')
        img2 = request.files.get('image2')

        if not img1 or not img2:
            return render_template('index.html', result="Please upload both images.")

        p1 = os.path.join(app.config['UPLOAD_FOLDER'], 'img1.jpg')
        p2 = os.path.join(app.config['UPLOAD_FOLDER'], 'img2.jpg')
        img1.save(p1)
        img2.save(p2)

        emb1 = extract_features(p1)
        emb2 = extract_features(p2)

        if emb1 is None or emb2 is None:
            return render_template('index.html', result="Face not detected in one or both images.")

        dist = euclidean_distance(emb1, emb2)
        cos = cosine_similarity(emb1, emb2)

        # Profile-aware thresholds
        if dist < 1.05 or cos > 0.55:
            result = f"Same Person (Distance={dist:.3f}, Cosine={cos:.3f} )"
        elif dist < 1.35:
            result = f"Moderate Similarity  (Distance={dist:.3f})"
        else:
            result = f"Different Persons  (Distance={dist:.3f})"

    return render_template('index.html', result=result)

# =======================
# Run app
# =======================
if __name__ == '__main__':
    app.run(debug=True)
