# Object Detection 

# 설치방법

이 프로젝트에서는 서버 용량 제한으로 인해 가상환경을 사용하지 않습니다. 대신 다음 단계를 따라 필요한 파일을 다운로드하고 환경을 설정합니다
## git clone
```bash
git clone https://github.com/boostcampaitech7/level2-objectdetection-cv-02.git
```
clone을 하고 나면 자동적으로 baseline code 는 다운받아져 있을 것이다. 이후 data를 다운받으면 된다.


## 데이터 및 코드 다운로드
```bash
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000325/data/data.tar.gz
# wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000325/data/20240902115340/code.tar.gz
```
## 압축 해제 및 정리
```bash
tar -zxvf data.tar.gz
# tar -zxvf code.tar.gz
rm code.tar.gz data.tar.gz
```

## ultralytics 세팅 마지막 준비
`dataset/labeling_make.ipynb` 를 켜서 작동시킨다.
train/images 폴더에 image들이 옮겨져 있고
train/labels 폴더가 새로 생기며 .txt파일들이 그 안에 생기면 완성이다.

이후 
`pip install ultralytics` 이후

터미널을 `/level2-objectdetection-cv-02/baseline/ultralytics ` 경로에 연다.

## yolo Train
`yolo train model=yolo11n.pt data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01` 

## yolo predict
`yolo predict model=yolo11n.pt source='your_path' imgsz=320`

## yolo test
-> `test.ipynb` 실행.

❗만약 dataset 경로를 못 찾는경우 
-> `yolo settings datasets_dir="../dataset"` 으로 바꾸고 다시 실행

## yolo special commands
yolo help   # 어떻게 yolo 를 사용하는지 간단한 명령어 모음 나옴
yolo checks # 본인이 yolo 를 사용할 수 있는 환경인지 나옴
yolo version # 지금 사용하고 있는 yolo version 나옴
yolo settings # 현재 yolo settings가 어떻게 되어 있는지 .setting.json에 있는 설정들 나옴
yolo cfg # 본인이 하고 있는 설정들이 나온다.
yolo copy-cfg


