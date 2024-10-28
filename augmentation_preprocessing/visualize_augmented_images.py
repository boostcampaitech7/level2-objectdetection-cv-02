import os
import json
import cv2
import matplotlib.pyplot as plt

# 데이터셋 경로 설정
dataset_path = "/data/ephemeral/home/level2-objectdetection-cv-02/dataset/"
augmented_annotations_file = os.path.join(dataset_path, "train_augmented.json")


# JSON 파일을 읽어오는 함수
def load_annotations(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# Augmented 이미지 시각화 함수
def visualize_augmented_images():
    data = load_annotations(augmented_annotations_file)
    annotations = data["annotations"]
    images = data["images"]
    display_count = 0  # Counter to keep track of displayed images

    for image_info in images:
        if display_count >= 5:
            break
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_path = os.path.join(dataset_path, "aug_" + file_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Augmented Image {display_count + 1}")
        plt.show()
        display_count += 1


if __name__ == "__main__":
    visualize_augmented_images()
