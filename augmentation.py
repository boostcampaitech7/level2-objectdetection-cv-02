import os
import json
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 데이터셋 경로 설정
dataset_path = "/home/user/dataset"
annotations_file = os.path.join(dataset_path, "train.json")
augmented_annotations_file = os.path.join(dataset_path, "train_augmented.json")


# JSON 파일을 읽어오는 함수
def load_annotations(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# Albumentations를 이용한 augmentation 함수
def augment_data(image, bboxes, category_ids):
    transform = A.Compose(
        [
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
            # A.Rotate(limit=30, p=0.5),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
    )
    augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    return augmented["image"], augmented["bboxes"], augmented["category_ids"]


# Augmentation된 이미지를 저장하고, 새로운 bounding box 좌표를 JSON 파일에 저장하는 함수
def save_augmented_data(
    image, bboxes, category_ids, original_image_path, annotations, image_id
):
    base_name = os.path.basename(original_image_path)
    new_image_path = os.path.join(dataset_path, "aug_" + base_name)
    cv2.imwrite(new_image_path, image)
    for bbox, category_id in zip(bboxes, category_ids):
        annotations.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "id": len(annotations) + 1,
            }
        )


# 메인 함수에서 전체 프로세스 실행
def main():
    data = load_annotations(annotations_file)
    annotations = data["annotations"]
    images = data["images"]
    for image_info in images:
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_path = os.path.join(os.path.join(dataset_path, "train"), file_name)
        image = cv2.imread(image_path)
        image_annotations = [ann for ann in annotations if ann["image_id"] == image_id]
        bboxes = [ann["bbox"] for ann in image_annotations]
        category_ids = [ann["category_id"] for ann in image_annotations]
        augmented_image, augmented_bboxes, augmented_category_ids = augment_data(
            image, bboxes, category_ids
        )
        save_augmented_data(
            augmented_image,
            augmented_bboxes,
            augmented_category_ids,
            image_path,
            annotations,
            image_id,
        )
    with open(augmented_annotations_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
