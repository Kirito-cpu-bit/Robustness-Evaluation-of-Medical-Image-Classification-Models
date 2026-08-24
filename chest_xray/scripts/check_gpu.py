"""Verify TensorFlow GPU (DirectML) availability and runtime placement."""
import sys
import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall


def main():
    print("=" * 60)
    print("Python:", sys.executable)
    print("TensorFlow:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("=" * 60)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"Detected GPU devices: {len(gpus)}")
    for index, gpu in enumerate(gpus):
        print(f"  [{index}] {gpu.name} ({gpu.device_type})")

    if not gpus:
        print("\nRESULT: No GPU detected. Scripts will fall back to CPU.")
        print("Fix: run with .venv-gpu\\Scripts\\python.exe")
        return 1

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Memory growth warning for {gpu.name}: {exc}")

    print("\nRunning placement benchmark on ResNet50 model ...")
    model_path = Path(__file__).resolve().parent.parent / "outputs" / "models" / "cnn_4_final.h5"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("GPU is available, but model benchmark was skipped.")
        print("\nRESULT: GPU detected and ready.")
        return 0

    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={"Recall": Recall, "Precision": Precision},
    )
    batch = tf.random.uniform((32, 224, 224, 3))

    with tf.device("/GPU:0"):
        _ = model(batch, training=False)
        start = time.time()
        for _ in range(5):
            _ = model(batch, training=False)
        gpu_seconds = time.time() - start

        with tf.GradientTape() as tape:
            tape.watch(batch)
            preds = model(batch, training=False)
            loss = tf.reduce_mean(preds)
        gradient = tape.gradient(loss, batch)

    with tf.device("/CPU:0"):
        start = time.time()
        for _ in range(5):
            _ = model(batch, training=False)
        cpu_seconds = time.time() - start

    print(f"GPU 5x predict: {gpu_seconds:.2f}s ({gpu_seconds / 5:.3f}s/batch)")
    print(f"CPU 5x predict: {cpu_seconds:.2f}s ({cpu_seconds / 5:.3f}s/batch)")
    print(f"Speedup (GPU/CPU): {cpu_seconds / gpu_seconds:.1f}x")
    print(f"Gradient tensor device: {gradient.device}")

    if "GPU" in gradient.device:
        print("\nRESULT: GPU is active (DirectML). Training/attacks use GPU:0.")
    else:
        print("\nRESULT: GPU detected but gradient ran on CPU. Check environment.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
