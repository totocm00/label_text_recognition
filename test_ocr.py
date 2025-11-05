# test_ocr.py
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='en')
res = ocr.ocr('test.jpg')
print(res[:2])
