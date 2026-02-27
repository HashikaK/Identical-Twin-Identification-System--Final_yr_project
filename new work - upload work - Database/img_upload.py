import os
import torch
import numpy as np
from flask import Flask, render_template, request
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageEnhance
from flask_sqlalchemy import SQLAlchemy
import pickle
from datetime import datetime

# -------------------- FLASK APP --------------------
app = Flask(__name__)

# -------------------- DATABASE CONFIG --------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///faces.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# -------------------- DATABASE MODEL --------------------
class Face(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    image_path = db.Column(db.String(200))
    embedding = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    model_version = db.Column(db.String(50), default="facenet_vggface2")

# -------------------- MODEL SETUP --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print(f"✅ Using device: {device}")

# -------------------- GLOBALS --------------------
known_faces = {}
SIMILARITY_THRESHOLD = 0.88  # Strict threshold for twins

# -------------------- PREPROCESS --------------------
def preprocess_image(img):
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    return img

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -------------------- LOAD FROM DATABASE --------------------
def load_faces_from_db():
    global known_faces
    known_faces = {}

    faces = Face.query.all()
    print("Total records in DB:", len(faces))

    for face in faces:
        emb = pickle.loads(face.embedding)

        if face.name not in known_faces:
            known_faces[face.name] = []

        known_faces[face.name].append(emb)

    print("✅ Loaded faces from database")

# -------------------- REGISTER PERSON --------------------
def register_person_folder(name, folder_path):

    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            image_path = os.path.join(folder_path, filename)

            img = Image.open(image_path).convert('RGB')
            img = preprocess_image(img)

            face_tensor = mtcnn(img)
            if face_tensor is None:
                print(f"❌ No face detected in {filename}")
                continue

            with torch.no_grad():
                emb = facenet(face_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()

            emb = emb / np.linalg.norm(emb)

            emb_bytes = pickle.dumps(emb)

            new_face = Face(
                name=name,
                image_path=image_path,
                embedding=emb_bytes,
                model_version="facenet_vggface2"
            )

            db.session.add(new_face)

    db.session.commit()
    print(f"✅ {name} saved to database")

# -------------------- IDENTIFICATION --------------------
def identify_person(file):

    img = preprocess_image(Image.open(file).convert('RGB'))
    face_tensor = mtcnn(img)

    if face_tensor is None:
        return "No Face Detected"

    with torch.no_grad():
        emb = facenet(face_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()

    emb = emb / np.linalg.norm(emb)

    best_name = None
    best_sim = -1.0
    second_best = -1.0

    for name, embedding_list in known_faces.items():

        sims = [cosine_similarity(emb, known_emb) for known_emb in embedding_list]
        sim = max(sims)

        print(f"🧠 Best similarity with {name}: {sim:.3f}")

        if sim > best_sim:
            second_best = best_sim
            best_sim = sim
            best_name = name
        elif sim > second_best:
            second_best = sim

    margin = best_sim - second_best

    if best_sim >= SIMILARITY_THRESHOLD:
        if margin < 0.04:
            return f"Ambiguous (Twin Similarity Too Close) — {best_sim:.3f}"
        return f"{best_name} ({best_sim:.3f})"
    else:
        return "Unknown Person"

# -------------------- ROUTE --------------------
@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        if 'image1' not in request.files or 'image2' not in request.files:
            return render_template('upload_index.html',
                                   result1="Upload Image 1",
                                   result2="Upload Image 2")

        file1 = request.files['image1']
        file2 = request.files['image2']

        name1 = identify_person(file1)
        name2 = identify_person(file2)

        return render_template('upload_index.html',
                               result1=name1,
                               result2=name2)

    return render_template('upload_index.html',
                           result1=None,
                           result2=None)

# -------------------- INITIALIZE DATABASE --------------------
with app.app_context():
    db.create_all()

    # FIRST TIME ONLY (then comment it)
    register_person_folder("Hashika", "uploads/hashika")
    register_person_folder("Hayanthika", "uploads/hayanthika")

    load_faces_from_db()

# -------------------- RUN --------------------
if __name__ == '__main__':
    app.run(debug=True)