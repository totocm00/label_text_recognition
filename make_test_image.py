# make_test_image.py
import cv2
import numpy as np

# 흰 배경 이미지
img = np.ones((300, 600, 3), dtype=np.uint8) * 255

# 테스트용 시리얼 넘버와 문구
texts = [
    "PCB-Serial: SN203948",
    "Voltage: 3.3V",
    "Board Ver: A1",
    "TEST OK"
]

# 폰트 세팅
font = cv2.FONT_HERSHEY_SIMPLEX

y = 60
for t in texts:
    cv2.putText(img, t, (50, y), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    y += 60

# 이미지 저장
cv2.imwrite("test.jpg", img)
print("✅ 'test.jpg' 생성 완료!")
