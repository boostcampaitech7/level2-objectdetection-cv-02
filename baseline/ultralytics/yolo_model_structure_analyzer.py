import torch

def analyze_yolo_model(model_path, model_name):
    print(f"\n{'='*50}")
    print(f"Analyzing {model_name}")
    print(f"{'='*50}")

    # Load .pt file
    model = torch.load(model_path)

    # Print model structure
    print(f"{model_name} structure:")
    print(model.keys())

    # Print model architecture
    if "model" in model:
        print(f"\n{model_name} architecture:")
        print(model["model"])
    else:
        print(f"\nWarning: 'model' key not found in {model_name}")

    # Additional analysis can be added here
    print(f"\n{model_name} analysis complete.")
    print(f"{'='*50}\n")

# Paths to different YOLO model versions
model_paths = {
    "YOLOv11n": "Project/level2-objectdetection-cv-02/baseline/ultralytics/yolo11n.pt",
    "YOLOv11s": "Project/level2-objectdetection-cv-02/baseline/ultralytics/yolo11s.pt",
    "YOLOv11m": "Project/level2-objectdetection-cv-02/baseline/ultralytics/yolo11m.pt",
    "YOLOv11l": "Project/level2-objectdetection-cv-02/baseline/ultralytics/yolo11l.pt",
    "YOLOv11x": "Project/level2-objectdetection-cv-02/baseline/ultralytics/yolo11x.pt"
}

# Analyze each model
for model_name, model_path in model_paths.items():
    try:
        analyze_yolo_model(model_path, model_name)
    except FileNotFoundError:
        print(f"Error: {model_name} file not found at {model_path}")
    except Exception as e:
        print(f"Error analyzing {model_name}: {str(e)}")

print("Analysis of all YOLO models complete.")