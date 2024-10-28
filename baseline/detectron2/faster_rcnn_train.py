import os
import copy
import torch
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import random
import numpy as np
import albumentations as A

# Register Dataset
try:
    register_coco_instances(
        "coco_trash_train", {}, "../../dataset/train.json", "../../dataset/"
    )
except AssertionError:
    pass

try:
    register_coco_instances(
        "coco_trash_test", {}, "../../dataset/test.json", "../../dataset/"
    )
except AssertionError:
    pass

MetadataCatalog.get("coco_trash_train").thing_classes = [
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]

# config 불러오기
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
)

# config 수정하기
cfg.DATASETS.TRAIN = ("coco_trash_train",)
cfg.DATASETS.TEST = ("coco_trash_test",)

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 15000
cfg.SOLVER.STEPS = (8000, 12000)
cfg.SOLVER.GAMMA = 0.005
cfg.SOLVER.CHECKPOINT_PERIOD = 3000

cfg.OUTPUT_DIR = "./output"

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

cfg.TEST.EVAL_PERIOD = 3000


def cutmix(image1, image2, bboxes1, bboxes2, category_ids1, category_ids2):
    h, w, _ = image1.shape
    cut_x = np.random.randint(w)
    cut_y = np.random.randint(h)

    new_image = image1.copy()
    new_image[cut_y:, cut_x:, :] = image2[cut_y:, cut_x:, :]

    # 두 번째 이미지의 경계 상자 좌표 변환
    new_bboxes2 = []
    for bbox in bboxes2:
        x, y, width, height = bbox
        new_bboxes2.append([x + cut_x, y + cut_y, width, height])

    new_bboxes = bboxes1 + new_bboxes2
    new_category_ids = category_ids1 + category_ids2

    return new_image, new_bboxes, new_category_ids


def mosaic(images, bboxes_list, category_ids_list):
    h, w, _ = images[0].shape
    new_image = np.zeros((2 * h, 2 * w, 3), dtype=images[0].dtype)

    new_image[:h, :w, :] = images[0]
    new_image[:h, w:, :] = images[1]
    new_image[h:, :w, :] = images[2]
    new_image[h:, w:, :] = images[3]

    # 각 이미지의 경계 상자 좌표 변환
    new_bboxes = []
    print(bboxes_list)
    bboxes_list_no_tuple = [
        [
            item
            for list2 in list1
            for item in (list2 if isinstance(list2, tuple) else [list2])
        ]
        for list1 in bboxes_list
    ]
    print(bboxes_list_no_tuple)
    for i, bbox in enumerate(bboxes_list_no_tuple):
        # for bbox in bbox_list:
        print(bbox)
        if i == 1:
            x, y, width, height = bbox
            new_bboxes.append([x + w, y, width, height])
        elif i == 2:
            x, y, width, height = bbox
            new_bboxes.append([x, y + h, width, height])
        elif i == 3:
            x, y, width, height = bbox
            new_bboxes.append([x + w, y + h, width, height])
        else:
            new_bboxes.append(bbox)

    new_category_ids = (
        category_ids_list[0]
        + category_ids_list[1]
        + category_ids_list[2]
        + category_ids_list[3]
    )

    return new_image, new_bboxes, new_category_ids


def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, [], image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    aug = A.Compose(
        [A.HorizontalFlip(p=1), A.VerticalFlip(p=1)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )

    aug_both = A.Compose(
        [A.HorizontalFlip(p=1), A.VerticalFlip(p=1)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )

    augmented_images = [image]
    augmented_annos = [annos]

    for transform in aug.transforms:
        augmented = transform(
            image=image,
            bboxes=[ann["bbox"] for ann in annos],
            category_ids=[ann["category_id"] for ann in annos],
        )
        augmented_images.append(augmented["image"])
        augmented_annos.append(
            [
                {"bbox": bbox, "category_id": category_id}
                for bbox, category_id in zip(
                    augmented["bboxes"], augmented["category_ids"]
                )
            ]
        )

    augmented_both = aug_both(
        image=image,
        bboxes=[ann["bbox"] for ann in annos],
        category_ids=[ann["category_id"] for ann in annos],
    )
    augmented_images.append(augmented_both["image"])
    augmented_annos.append(
        [
            {"bbox": bbox, "category_id": category_id}
            for bbox, category_id in zip(
                augmented_both["bboxes"], augmented_both["category_ids"]
            )
        ]
    )

    cutmix_images = []
    mosaic_images = []

    for _ in range(1000):
        img1, annos1 = random.choice(list(zip(augmented_images, augmented_annos)))
        img2, annos2 = random.choice(list(zip(augmented_images, augmented_annos)))

        bboxes1 = [ann["bbox"] for ann in annos1]
        category_ids1 = [ann["category_id"] for ann in annos1]
        bboxes2 = [ann["bbox"] for ann in annos2]
        category_ids2 = [ann["category_id"] for ann in annos2]

        cutmix_image, cutmix_bboxes, cutmix_category_ids = cutmix(
            img1, img2, bboxes1, bboxes2, category_ids1, category_ids2
        )
        cutmix_images.append((cutmix_image, cutmix_bboxes, cutmix_category_ids))

        # # 이미지와 주석을 함께 무작위로 선택하여 매핑을 유지합니다.
        # mosaic_pairs = [
        #     random.choice(list(zip(augmented_images, augmented_annos)))
        #     for _ in range(4)
        # ]
        # mosaic_imgs, mosaic_annos = zip(*mosaic_pairs)

        # 이미지와 주석을 함께 무작위로 선택하지 않고, 0번째 이미지부터 3번째 이미지까지 선택하여 매핑을 유지합니다.
        # mosaic_pairs = list(zip(augmented_images[:4], augmented_annos[:4]))
        # mosaic_imgs, mosaic_annos = zip(*mosaic_pairs)

        mosaic_imgs = augmented_images[1:5]
        mosaic_annos = augmented_annos[1:5]

        # 각 이미지의 bbox 리스트를 개별적으로 생성
        mosaic_bboxes = [[ann["bbox"] for ann in annos] for annos in mosaic_annos]
        mosaic_category_ids = [
            [ann["category_id"] for ann in annos] for annos in mosaic_annos
        ]

        mosaic_image, mosaic_bboxes, mosaic_category_ids = mosaic(
            mosaic_imgs, mosaic_bboxes, mosaic_category_ids
        )
        mosaic_images.append([mosaic_image, mosaic_bboxes, mosaic_category_ids])

    new_dataset_dicts = []

    for cutmix_image, cutmix_bboxes, cutmix_category_ids in cutmix_images:
        cutmix_dataset_dict = copy.deepcopy(dataset_dict)
        cutmix_dataset_dict["image"] = torch.as_tensor(
            cutmix_image.transpose(2, 0, 1).astype("float32")
        )
        cutmix_annos = [
            {"bbox": bbox, "category_id": category_id}
            for bbox, category_id in zip(cutmix_bboxes, cutmix_category_ids)
        ]
        cutmix_instances = utils.annotations_to_instances(
            cutmix_annos, cutmix_image.shape[:2]
        )
        cutmix_dataset_dict["instances"] = utils.filter_empty_instances(
            cutmix_instances
        )
        new_dataset_dicts.append(cutmix_dataset_dict)

    for mosaic_image, mosaic_bboxes, mosaic_category_ids in mosaic_images:
        mosaic_dataset_dict = copy.deepcopy(dataset_dict)
        mosaic_dataset_dict["image"] = torch.as_tensor(
            mosaic_image.transpose(2, 0, 1).astype("float32")
        )
        mosaic_annos = [
            {"bbox": bbox, "category_id": category_id}
            for bbox, category_id in zip(mosaic_bboxes, mosaic_category_ids)
        ]
        mosaic_instances = utils.annotations_to_instances(
            mosaic_annos, mosaic_image.shape[:2]
        )
        mosaic_dataset_dict["instances"] = utils.filter_empty_instances(
            mosaic_instances
        )
        new_dataset_dicts.append(mosaic_dataset_dict)

    return new_dataset_dicts


class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(cfg, mapper=MyMapper, sampler=sampler)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("./output_eval", exist_ok=True)
            output_folder = "./output_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
