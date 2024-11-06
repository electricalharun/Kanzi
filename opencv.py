import cv2
import numpy as np

# Kamerayı aç
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gri görüntüyü ikili (binary) hale getirmek için bir eşik uygulayın
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # Beyaz kağıt için uygun eşik değeri

    # Konturları bul
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 5000:  # Kağıdın boyutunu kontrol et
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Kutu çiz
            cv2.putText(frame, 'Beyaz Kağıt', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Beyaz Kağıt Takibi', frame)  # Sonuç görüntüsünü göster

    if cv2.waitKey(1) & 0xFF == ord('b'):  # 'q' tuşuna basılırsa çık
        break

cap.release()  # Kamerayı kapat
cv2.destroyAllWindows()  # Tüm pencereleri kapat
