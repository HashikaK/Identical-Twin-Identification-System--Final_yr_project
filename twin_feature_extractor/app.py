import os
import io
import torch
import secrets
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs_encrypted'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# FaceNet + MTCNN setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print(f"Using device: {device}")

# AES encryption helpers
KDF_SALT_SIZE = 16
KDF_ITERATIONS = 200_000
AES_KEY_SIZE = 32
AES_NONCE_SIZE = 12

def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=AES_KEY_SIZE,
        salt=salt,
        iterations=KDF_ITERATIONS,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

def encrypt_bytes(plaintext: bytes, password: str) -> bytes:
    salt = secrets.token_bytes(KDF_SALT_SIZE)
    key = derive_key(password, salt)
    nonce = secrets.token_bytes(AES_NONCE_SIZE)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return salt + nonce + ciphertext

def decrypt_bytes(blob: bytes, password: str) -> bytes:
    salt = blob[:KDF_SALT_SIZE]
    nonce = blob[KDF_SALT_SIZE:KDF_SALT_SIZE + AES_NONCE_SIZE]
    ciphertext = blob[KDF_SALT_SIZE + AES_NONCE_SIZE:]
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)

# Embedding serialization
def serialize_embedding(emb: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, emb)
    return bio.getvalue()

def deserialize_embedding(data: bytes) -> np.ndarray:
    bio = io.BytesIO(data)
    bio.seek(0)
    return np.load(bio, allow_pickle=False)

# Feature extraction
def extract_features(image_path: str):
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return None
    with torch.no_grad():
        emb = facenet(face.unsqueeze(0).to(device))
    emb = emb.squeeze().cpu().numpy()
    return emb / np.linalg.norm(emb)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Flask Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        password = request.form['password']
        img1 = request.files['image1']
        img2 = request.files['image2']

        if not img1 or not img2 or not password:
            return render_template(
                'index.html',
                result="Please upload both images and enter a password."
            )

        # Save uploaded images
        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img1.jpg')
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img2.jpg')
        img1.save(img1_path)
        img2.save(img2_path)

        # Extract features
        emb1 = extract_features(img1_path)
        emb2 = extract_features(img2_path)

        if emb1 is None or emb2 is None:
            return render_template(
                'index.html',
                result="Face not detected in one or both images."
            )

        # Encrypt embeddings
        enc1 = encrypt_bytes(serialize_embedding(emb1), password)
        enc2 = encrypt_bytes(serialize_embedding(emb2), password)

        enc1_path = os.path.join(app.config['OUTPUT_FOLDER'], 'img1.enc')
        enc2_path = os.path.join(app.config['OUTPUT_FOLDER'], 'img2.enc')

        with open(enc1_path, 'wb') as f:
            f.write(enc1)
        with open(enc2_path, 'wb') as f:
            f.write(enc2)

        # Decrypt for comparison
        emb1_dec = deserialize_embedding(decrypt_bytes(enc1, password))
        emb2_dec = deserialize_embedding(decrypt_bytes(enc2, password))

        # Compute similarity
        similarity = cosine_similarity(emb1_dec, emb2_dec)

        if similarity > 0.8:
            message = f"Highly similar (Similarity: {similarity:.4f})"
        elif similarity > 0.5:
            message = f"Moderate similarity (Similarity: {similarity:.4f})"
        else:
            message = f"Different persons (Similarity: {similarity:.4f})"

        return render_template('index.html', result=message)

    return render_template('index.html', result=None)

# Run App
if __name__ == '__main__':
    app.run(debug=True)
