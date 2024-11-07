import os
import random
import subprocess

# 모델 경로 설정
model_path = "/home/work/Project/level2-objectdetection-cv-02/baseline/ultralytics/recycling/experiment2/weights/best.pt"

# 이미지가 있는 디렉토리 경로 설정
image_dir = "/home/work/Project/level2-objectdetection-cv-02/dataset/test/"

# 결과를 저장할 디렉토리 설정
output_dir = "/home/work/Project/level2-objectdetection-cv-02/baseline/ultralytics/predict_results"

# 디렉토리에서 모든 jpg 파일 리스트 가져오기
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# 랜덤으로 10개의 이미지 선택 (또는 원하는 개수로 조정)
num_samples = min(10, len(image_files))
random_samples = random.sample(image_files, num_samples)

# 선택된 이미지 경로를 저장할 임시 파일 생성
with open('temp_image_list.txt', 'w') as f: #w 모드는 쓰기모드이며, 파일이 없으면 새로 만듦
    for image_file in random_samples:
        f.write(os.path.join(image_dir, image_file) + '\n') #이미지파일의 절대경로를 나타내며 이를 temp_image.txt에 넣음

# YOLO 예측 실행
command = f"yolo predict model='{model_path}' source='temp_image_list.txt' project='{output_dir}' name='batch_predict'"
print(f"Executing: {command}")
subprocess.run(command, shell=True)

# 임시 파일 삭제
os.remove('temp_image_list.txt')

print(f"Results saved to {os.path.join(output_dir, 'batch_predict')}")