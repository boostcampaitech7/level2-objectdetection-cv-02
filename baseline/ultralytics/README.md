# Object Detection 

# 설치방법

이 프로젝트에서는 서버 용량 제한으로 인해 가상환경을 사용하지 않습니다. 대신 다음 단계를 따라 필요한 파일을 다운로드하고 환경을 설정합니다

## git clone
```bash
git clone https://github.com/boostcampaitech7/level2-objectdetection-cv-02.git
```
clone을 하고 나면 자동적으로 baseline code 는 다운받아져 있을 것이다. 이후 data를 다운받으면 된다.


## 데이터 및 다운로드
```bash
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000325/data/data.tar.gz

```
## 압축 해제 및 정리
```bash
tar -zxvf data.tar.gz
rm  data.tar.gz
```

## ultralytics 세팅 마지막 준비
`dataset/labeling_make.ipynb` 를 켜서 작동시킨다.
train/images 폴더에 image들이 옮겨져 있고
train/labels 폴더가 새로 생기며 .txt파일들이 그 안에 생기면 완성이다.

이후 
`pip install ultralytics` 이후

터미널을 `/level2-objectdetection-cv-02/baseline/ultralytics ` 경로에서 열어야 합니다.

## yolo config
`baseline/ultralytics` 폴더에 들어가면 `config.yaml`이 있습니다. 이를 이용하여 하이퍼파라미터 및 본인이 원하는 모델을 적용할 수 있습니다.

> 어떠한 파라미터들이 있는지 알고 싶다면은,`level2-objectdetection-cv-02/baseline/ultralytics/ultralytics/cfg/defalut.yaml`  를 열어보세요! 다양한 파라미터들이 있음을 알 수 있습니다

자세한 config 설명은 :https://docs.ultralytics.com/usage/cfg/

## CLI interface Usage
yolo TASK MODE ARGS

- TASK (optional) is one of [detect, segment, classify, pose, obb]

- MODE (required) is one of [train, val, predict, export, track, benchmark]

- ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.

## yolo Train
`yolo train cfg=config.yaml` 

## yolo predict
`yolo predict cfg=config.yaml model=best.pt source='your_path' imgsz=320`

## yolo test
-> `test.ipynb` 실행.

❗만약 dataset 경로를 못 찾는경우 
-> `yolo settings datasets_dir="../dataset"` 으로 바꾸고 다시 실행


## yolo special commands
- `yolo help`   # 어떻게 yolo 를 사용하는지 간단한 명령어 모음 나옴
- `yolo checks` # 본인이 yolo 를 사용할 수 있는 환경인지 나옴
- `yolo version` # 지금 사용하고 있는 yolo version 나옴
- `yolo settings` # 현재 yolo settings가 어떻게 되어 있는지 .setting.json에 있는 설정들 나옴
- `yolo cfg` # 본인이 하고 있는 설정들이 나온다.
- `yolo copy-cfg` # 현재 config설정들이랑 똑같은 것을 만든다.

## yolo tracking
`yolo track source="path/to/video"`

## yolo hub
위 프로젝트는 ultralytics에서 제공하는 hub를 통해 프로젝트의 진행상황을 모니터링 합니다. 

## yolov11 모델들의 다양한 구조에 대해서 궁금하다면?
`level2-objectdetection-cv-02/baseline/ultralytics/ultralytics/cfg/models/11/yolo11.yaml`<- 여기 yaml파일을 열어보면 됩니다.

![alt text](image.png) 
위 이미지처럼 잘 나와있다..!