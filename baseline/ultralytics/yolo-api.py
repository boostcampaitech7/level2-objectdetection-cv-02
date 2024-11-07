import json
import requests
import os

# Run inference on an image
url = "https://predict.ultralytics.com"
headers = {"x-api-key": os.getenv("ULTRALYTICS_API_KEY")} #<- 본인의 ultralytic hub에 가입하여 api key를 입력하세요!
data = {"model": "https://hub.ultralytics.com/models/7wzkDSKNMcwkPTs8ZVJC", "imgsz": 640, "conf": 0.25, "iou": 0.45}
with open("/home/work/Project/level2-objectdetection-cv-02/baseline/ultralytics/ultralytics/assets/bus.jpg", "rb") as f:
	response = requests.post(url, headers=headers, data=data, files={"file": f})

# Check for successful response
response.raise_for_status()

# Print inference results
print(json.dumps(response.json(), indent=2))
results =response.json()
print(results["images"]) # predict 한 모든 dictionary result 출력.