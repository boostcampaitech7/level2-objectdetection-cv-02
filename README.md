# Object Detection 

# 설치방법

이 프로젝트에서는 서버 용량 제한으로 인해 가상환경을 사용하지 않습니다. 대신 다음 단계를 따라 필요한 파일을 다운로드하고 환경을 설정합니다

## 데이터 및 코드 다운로드
```bash
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000325/data/data.tar.gz
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000325/data/20240902115340/code.tar.gz
```
## 압축 해제 및 정리
```bash
tar -zxvf data.tar.gz
tar -zxvf code.tar.gz
rm code.tar.gz data.tar.gz
```
## Detectron2 설치
```bash
cd baseline
python -m pip install -e detectron2  #detectron2 에 있는 setup.py, setup.cfg를 이용하여 필요한 패키지들을 자동으로 다운받아줍니다.
```

이제 `faster_rcnn_train.ipynb` 실행 후에, `faster_rcnn_inference.ipynb`으로 테스트하면 됩니다.
-> 후에 `.ipynb`로 하는 것이 아니라, `py`로 고쳐서 해보장.
