import os
import json
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# 데이터셋 경로 설정
dataset_path = "/data/ephemeral/home/level2-objectdetection-cv-02/dataset/"
annotations_file = os.path.join(dataset_path, "train.json")
augmented_annotations_file = os.path.join(dataset_path, "train_augmented.json")
augmented_images_path = os.path.join(dataset_path, "augmented_train")

# augmented_train 폴더가 없으면 생성
if not os.path.exists(augmented_images_path):
    os.makedirs(augmented_images_path)


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
            A.VerticalFlip(p=0.5),
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
    new_image_path = os.path.join(augmented_images_path, "aug_" + base_name)

    # PyTorch 텐서를 NumPy 배열로 변환
    image_np = image.permute(1, 2, 0).cpu().numpy()

    cv2.imwrite(new_image_path, image_np)
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
        image_path = os.path.join(dataset_path, file_name)
        print(f"Trying to read image from: {image_path}")  # 경로 출력
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
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


# import os
# import json
# import cv2
# import numpy as np

# # 데이터셋 경로 설정
# dataset_path = "/data/ephemeral/home/level2-objectdetection-cv-02/dataset/"
# annotations_file = os.path.join(dataset_path, "train.json")
# augmented_annotations_file = os.path.join(dataset_path, "train_augmented.json")
# augmented_images_path = os.path.join(dataset_path, "augmented_train")

# # augmented_train 폴더가 없으면 생성
# if not os.path.exists(augmented_images_path):
#     os.makedirs(augmented_images_path)

# # JSON 파일을 읽어오는 함수
# def load_annotations(file_path):
#     with open(file_path, "r") as f:
#         data = json.load(f)
#     return data

# # CutMix 함수
# def cutmix(image1, image2, bboxes1, bboxes2, category_ids1, category_ids2):
#     h, w, _ = image1.shape
#     cut_x = np.random.randint(w)
#     cut_y = np.random.randint(h)

#     new_image = image1.copy()
#     new_image[cut_y:, cut_x:, :] = image2[cut_y:, cut_x:, :]

#     new_bboxes = bboxes1 + bboxes2
#     new_category_ids = category_ids1 + category_ids2

#     return new_image, new_bboxes, new_category_ids

# # Mosaic 함수
# def mosaic(images, bboxes_list, category_ids_list):
#     h, w, _ = images[0].shape
#     new_image = np.zeros((2 * h, 2 * w, 3), dtype=images[0].dtype)

#     new_image[:h, :w, :] = images[0]
#     new_image[:h, w:, :] = images[1]
#     new_image[h:, :w, :] = images[2]
#     new_image[h:, w:, :] = images[3]

#     new_bboxes = bboxes_list[0] + bboxes_list[1] + bboxes_list[2] + bboxes_list[3]
#     new_category_ids = category_ids_list[0] + category_ids_list[1] + category_ids_list[2] + category_ids_list[3]

#     return new_image, new_bboxes, new_category_ids

# # Augmentation된 이미지를 저장하고, 새로운 bounding box 좌표를 JSON 파일에 저장하는 함수
# def save_augmented_data(image, bboxes, category_ids, original_image_path, annotations, image_id):
#     base_name = os.path.basename(original_image_path)
#     new_image_path = os.path.join(augmented_images_path, "aug_" + base_name)

#     cv2.imwrite(new_image_path, image)
#     for bbox, category_id in zip(bboxes, category_ids):
#         annotations.append(
#             {
#                 "image_id": image_id,
#                 "category_id": category_id,
#                 "bbox": bbox,
#                 "area": bbox[2] * bbox[3],
#                 "iscrowd": 0,
#                 "id": len(annotations) + 1,
#             }
#         )

# # 메인 함수에서 전체 프로세스 실행
# def main():
#     data = load_annotations(annotations_file)
#     annotations = data["annotations"]
#     images = data["images"]
#     num_images = len(images)

#     for i in range(0, num_images, 2):
#         if i + 1 >= num_images:
#             break

#         image_info1 = images[i]
#         image_info2 = images[i + 1]

#         image_id1 = image_info1["id"]
#         image_id2 = image_info2["id"]

#         file_name1 = image_info1["file_name"]
#         file_name2 = image_info2["file_name"]

#         image_path1 = os.path.join(dataset_path, "train", file_name1)
#         image_path2 = os.path.join(dataset_path, "train", file_name2)

#         image1 = cv2.imread(image_path1)
#         image2 = cv2.imread(image_path2)

#         if image1 is None or image2 is None:
#             print(f"Warning: Could not read images {image_path1} or {image_path2}")
#             continue

#         image_annotations1 = [ann for ann in annotations if ann["image_id"] == image_id1]
#         image_annotations2 = [ann for ann in annotations if ann["image_id"] == image_id2]

#         bboxes1 = [ann["bbox"] for ann in image_annotations1]
#         bboxes2 = [ann["bbox"] for ann in image_annotations2]

#         category_ids1 = [ann["category_id"] for ann in image_annotations1]
#         category_ids2 = [ann["category_id"] for ann in image_annotations2]

#         # CutMix 적용
#         cutmix_image, cutmix_bboxes, cutmix_category_ids = cutmix(
#             image1, image2, bboxes1, bboxes2, category_ids1, category_ids2
#         )
#         save_augmented_data(
#             cutmix_image, cutmix_bboxes, cutmix_category_ids, image_path1, annotations, image_id1
#         )

#         # Mosaic 적용
#         if i + 3 < num_images:
#             image_info3 = images[i + 2]
#             image_info4 = images[i + 3]

#             image_id3 = image_info3["id"]
#             image_id4 = image_info4["id"]

#             file_name3 = image_info3["file_name"]
#             file_name4 = image_info4["file_name"]

#             image_path3 = os.path.join(dataset_path, "train", file_name3)
#             image_path4 = os.path.join(dataset_path, "train", file_name4)

#             image3 = cv2.imread(image_path3)
#             image4 = cv2.imread(image_path4)

#             if image3 is None or image4 is None:
#                 print(f"Warning: Could not read images {image_path3} or {image_path4}")
#                 continue

#             image_annotations3 = [ann for ann in annotations if ann["image_id"] == image_id3]
#             image_annotations4 = [ann for ann in annotations if ann["image_id"] == image_id4]

#             bboxes3 = [ann["bbox"] for ann in image_annotations3]
#             bboxes4 = [ann["bbox"] for ann in image_annotations4]

#             category_ids3 = [ann["category_id"] for ann in image_annotations3]
#             category_ids4 = [ann["category_id"] for ann in image_annotations4]

#             mosaic_image, mosaic_bboxes, mosaic_category_ids = mosaic(
#                 [image1, image2, image3, image4],
#                 [bboxes1, bboxes2, bboxes3, bboxes4],
#                 [category_ids1, category_ids2, category_ids3, category_ids4]
#             )
#             save_augmented_data(
#                 mosaic_image, mosaic_bboxes, mosaic_category_ids, image_path1, annotations, image_id1
#             )

#     with open(augmented_annotations_file, "w") as f:
#         json.dump(data, f)

# if __name__ == "__main__":
#     main()
