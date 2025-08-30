import cv2
import easyocr

class LicensePlateReader:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def read_license_plate(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        license_plate_crop = frame[y1:y2, x1:x2]
        if license_plate_crop.size == 0:
            return None
        gray_plate = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray_plate)
        text = ' '.join([result[1] for result in results])
        return text.strip() if text else None
