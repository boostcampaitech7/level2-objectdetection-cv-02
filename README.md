## Overview
목표: 이미지속 쓰레기를 분류하는 모델 제작

데이터셋: 다양한 크기의 쓰레기 이미지 (1024x1024)

평가지표: mAP50

<br><br>

## Member  
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/kim-minsol"><img height="110px" src="https://avatars.githubusercontent.com/u/81224613?v=4"/></a>
            <br />
            <a href="https://github.com/kim-minsol"><strong>김민솔</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/joonhyunkim1"><img height="110px"  src="https://avatars.githubusercontent.com/u/141805564?v=4"/></a>
              <br />
              <a href="https://github.com/joonhyunkim1"><strong>김준현</strong></a>
              <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/sweetie-orange"><img height="110px"  src="https://avatars.githubusercontent.com/u/97962649?v=4"/></a>
              <br />
              <a href="https://github.com/sweetie-orange"><strong>김현진</strong></a>
              <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/0seoYun"><img height="110px"  src="https://avatars.githubusercontent.com/u/102219161?v=4"/></a>
              <br />
              <a href="https://github.com/0seoYun"><strong>윤영서</strong></a>
              <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/2JAE22"><img height="110px"  src="https://avatars.githubusercontent.com/u/87936538?v=4"/></a>
              <br />
              <a href="https://github.com/2JAE22"><strong>이재건</strong></a>
              <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/Gwonee"><img height="110px"  src="https://avatars.githubusercontent.com/u/125177607?v=4"/></a>
              <br />
              <a href="https://github.com/Gwonee"><strong>정권희</strong></a>
              <br />
        </td>
    </tr>
</table>  

<br>

### 역할

|팀원|역할|
|-----|---|
|김민솔| 모델링(MMDetection) |
|김준현| Data augmentationDetectron2), Ensemble |
|김현진| EDA, 모델링(MMDetction) |
|윤영서| Augmentation, Project Manager |
|이재건| Yolo 모델 적용 |
|정권희| 모델링(Detectron2), EnsembleData |

<br><br>


## Methods

|분류|내용|
  |-----|---|
  |Models| **MMDetection**: CascadeRCNN, EfficientNet, DINO, DETR <br> **Detectron2**: Faster-RCNN, Retinanet, Cascade-rcnn <br>  **YOLO**: YOLOv11 ( s, l, xl) |
  |Augmentation| - Mosaic <br> - RandomCrop <br> - MixUp <br> - RandomBrightness <br> - RandomCropWithMinIoU <br> - ResizeWithAspectRatioClipping|
  |Ensemble|- Soft-NMS <br> - WBF |


<br><br>

## Result
최종본은 가장 성능이 좋았던 모델들을 앙상블하여 구함.
- cascade-rcnn-swin-L
- DINO-base
- DINO-resnet50

### Score
||Public Dataset|Private Dataset|
|---|-----|---|
|Score| 0.6714 | 0.6558 |


