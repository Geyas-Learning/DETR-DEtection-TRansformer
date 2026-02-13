"""
=========================================================
MAIN PIPELINE — DETR for Spacecraft Detection
=========================================================
"""
import importlib.util
import os
from data_utils import preprocess_original_csv, validate_image_paths, DATA_ROOT



MODEL_NAME = "DR152_pipeline"
MODEL_FILE_PATH = f"detection_detr/{MODEL_NAME}.py"


def ensure_csv(csv_path, raw_csv_name, split):
    """Ensure processed CSV exists; rebuild if missing or corrupted."""
    if not os.path.exists(csv_path):
        print(f"[WARN] Missing processed CSV → rebuilding: {csv_path}")
        preprocess_original_csv(raw_csv_name, csv_path, split)
        return
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline()
            if "," not in header:
                raise ValueError("Not a valid CSV header")
    except Exception:
        print(f"[WARN] Corrupted CSV → rebuilding: {csv_path}")
        preprocess_original_csv(raw_csv_name, csv_path, split)


def main():
    print(f"\nStarting Spacecraft Detection Pipeline using model: {MODEL_NAME}")

    # ------------------------------------------------------------
    # Prepare CSVs
    # ------------------------------------------------------------
    train_csv = os.path.join(DATA_ROOT, "train_processed.csv")
    val_csv = os.path.join(DATA_ROOT, "val_processed.csv")

    ensure_csv(train_csv, "train.csv", "train")
    ensure_csv(val_csv, "val.csv", "val")
    validate_image_paths(train_csv)
    validate_image_paths(val_csv)

    # ------------------------------------------------------------
    # Load model dynamically
    # ------------------------------------------------------------
    print(f"\n[INFO] Loading model code from: {MODEL_FILE_PATH}")
    spec = importlib.util.spec_from_file_location("model_lib", MODEL_FILE_PATH)
    if spec is None:
        print(f"ERROR: Could not find model file at: {MODEL_FILE_PATH}")
        return
    model_lib = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(model_lib)
    except Exception as e:
        print(f"ERROR: Failed to import model module '{MODEL_NAME}'.\nDetails: {e}")
        return

    # ------------------------------------------------------------
    # Config
    # ------------------------------------------------------------
    config = {
        "epochs": 100,
        "batch": 16,
        "imgsz": 224,
        "project_dir": "./DR50_12layers/train_detr",
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_images_full_path": "./test/images",
        "run_name": MODEL_NAME
    }
    os.makedirs(config["project_dir"], exist_ok=True)

    # ------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------
    print("\n[PIPELINE] Beginning model operations...")

    if hasattr(model_lib, "train_model"):
        print("[STEP 1/3] Training model...")
        model_lib.train_model(config)
    else:
        print("train_model not found.")

    if hasattr(model_lib, "test_model"):
        print("\n[STEP 2/3] Running inference...")
        model_lib.test_model(config)
    else:
        print("test_model not found.")

    if hasattr(model_lib, "visualize_predictions"):
        print("\n[STEP 3/3] 🖼️ Visualizing predictions...")
        model_lib.visualize_predictions(config)
    else:
        print("visualize_predictions not found.")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()