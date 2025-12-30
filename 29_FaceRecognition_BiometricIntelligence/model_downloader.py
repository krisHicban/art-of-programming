"""
Model Downloader Utility for Face Recognition Scripts
Auto-downloads required dlib models if they don't exist
"""
import os
import bz2
import urllib.request
from pathlib import Path

# Model URLs
MODELS = {
    'shape_predictor_68_face_landmarks.dat': {
        'url': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
        'compressed': True
    },
    'dlib_face_recognition_resnet_model_v1.dat': {
        'url': 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
        'compressed': True
    },
    'mmod_human_face_detector.dat': {
        'url': 'http://dlib.net/files/mmod_human_face_detector.dat.bz2',
        'compressed': True
    }
}

def download_model(model_name, download_dir='.'):
    """
    Download and extract a dlib model if it doesn't exist

    Args:
        model_name: Name of the model file (e.g., 'shape_predictor_68_face_landmarks.dat')
        download_dir: Directory to save the model (default: current directory)

    Returns:
        Path to the model file
    """
    model_path = Path(download_dir) / model_name

    # Check if model already exists
    if model_path.exists():
        print(f"✅ Model already exists: {model_name}")
        return str(model_path)

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model_info = MODELS[model_name]
    url = model_info['url']
    compressed_path = model_path.with_suffix('.dat.bz2')

    print(f"\n📥 Downloading {model_name}...")
    print(f"   URL: {url}")
    print(f"   This may take a few minutes depending on your connection...")

    try:
        # Download compressed file
        urllib.request.urlretrieve(url, compressed_path)
        print(f"✅ Downloaded: {compressed_path.name}")

        # Decompress if needed
        if model_info['compressed']:
            print(f"📦 Decompressing {compressed_path.name}...")
            with bz2.open(compressed_path, 'rb') as f_in:
                with open(model_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            # Remove compressed file
            compressed_path.unlink()
            print(f"✅ Extracted: {model_name}")

        print(f"✅ Model ready: {model_path}")
        return str(model_path)

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        # Clean up partial downloads
        if compressed_path.exists():
            compressed_path.unlink()
        if model_path.exists():
            model_path.unlink()
        raise

def ensure_models(*model_names, download_dir='.'):
    """
    Ensure multiple models are downloaded

    Args:
        *model_names: Variable number of model names
        download_dir: Directory to save models

    Returns:
        Dictionary mapping model names to their paths
    """
    model_paths = {}

    print("\n🔍 Checking required models...")
    for model_name in model_names:
        model_paths[model_name] = download_model(model_name, download_dir)

    print("\n✅ All required models are ready!")
    return model_paths

if __name__ == "__main__":
    # Test the downloader
    print("Testing model downloader...")
    ensure_models(
        'shape_predictor_68_face_landmarks.dat',
        'dlib_face_recognition_resnet_model_v1.dat'
    )
