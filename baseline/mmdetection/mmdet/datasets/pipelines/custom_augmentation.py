import random
import cv2
import numpy as np
from mmdet.datasets.pipelines import PIPELINES


@PIPELINES.register_module()
class RandomBrightness(object):
    '''
    이미지의 밝기를 랜덤으로 변화시키는 기법입니다.
    '''
    def __init__(self, brightness_delta=32):
        self.brightness_delta = brightness_delta

    def __call__(self, results):
        img = results['img']
        if random.randint(0, 1):
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img = np.clip(img + delta, 0, 255)
        results['img'] = img
        return results


@PIPELINES.register_module()
class RandomCropWithMinIoU(object):
    '''
    작은 객체를 감안하여 이미지의 일부분을 무작위로 자르되, 
    객체와의 교차 영역이 일정 비율 이상이 되도록 하는 기법입니다. 
    이를 통해 작은 객체가 잘리지 않고 학습에 활용될 수 있습니다.
    이 증강법은 작은 객체가 이미지의 잘린 부분에 포함될 수 있도록 조정합니다.
    '''
    def __init__(self, min_iou=0.5):
        self.min_iou = min_iou

    def __call__(self, results):
        img = results['img']
        bboxes = results['gt_bboxes']

        # 이미지의 일정 부분을 자름
        h, w, _ = img.shape
        new_h, new_w = random.randint(int(h * 0.6), h), random.randint(int(w * 0.6), w)
        top, left = random.randint(0, h - new_h), random.randint(0, w - new_w)
        
        cropped_img = img[top:top + new_h, left:left + new_w]

        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # IoU 계산하여 일정 비율 이상인 bbox만 남김
            if x2 > left and x1 < left + new_w and y2 > top and y1 < top + new_h:
                new_bboxes.append([max(x1 - left, 0), max(y1 - top, 0),
                                   min(x2 - left, new_w), min(y2 - top, new_h)])
        
        if new_bboxes:
            results['img'] = cropped_img
            results['gt_bboxes'] = np.array(new_bboxes)
        
        return results
    

@PIPELINES.register_module()
class ResizeWithAspectRatioClipping(object):
    '''
    작은 객체가 지나치게 왜곡되지 않도록 비율을 유지하면서 객체를 확대하는 기법입니다. 
    특히 작은 객체는 확대되면서 이미지에서 더욱 두드러질 수 있습니다.
    이 기법은 작은 객체들이 이미지에서 너무 작게 나타나지 않도록 확대하며, 
    비율을 유지하기 때문에 왜곡 없이 학습에 도움이 될 수 있습니다.
    '''
    def __init__(self, img_scale=(1024, 1024), keep_ratio=True):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def __call__(self, results):
        img = results['img']
        h, w, _ = img.shape
        scale = min(self.img_scale[0] / w, self.img_scale[1] / h)
        if self.keep_ratio:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            # 비율에 맞춰 잘라내기
            padded_img = np.zeros((self.img_scale[1], self.img_scale[0], 3), dtype=img.dtype)
            padded_img[:new_h, :new_w, :] = img
            results['img'] = padded_img
        else:
            results['img'] = cv2.resize(img, self.img_scale)
        return results
    

@PIPELINES.register_module()
class RandomErasingForSmallObjects(object):
    '''
    작은 객체 위에 작은 패치를 무작위로 덧대는 기법입니다. 
    작은 객체가 가려지는 상황을 학습시킴으로써 모델이 강건성을 가지도록 유도할 수 있습니다.
    '''
    def __init__(self, erasing_prob=0.5, min_area_ratio=0.01, max_area_ratio=0.05):
        self.erasing_prob = erasing_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def __call__(self, results):
        img = results['img']
        h, w, _ = img.shape
        if random.random() < self.erasing_prob:
            erase_h = random.randint(int(h * self.min_area_ratio), int(h * self.max_area_ratio))
            erase_w = random.randint(int(w * self.min_area_ratio), int(w * self.max_area_ratio))
            x = random.randint(0, w - erase_w)
            y = random.randint(0, h - erase_h)

            # 무작위 색으로 지우기
            img[y:y + erase_h, x:x + erase_w, :] = np.random.randint(0, 256, (erase_h, erase_w, 3), dtype=np.uint8)

        results['img'] = img
        return results