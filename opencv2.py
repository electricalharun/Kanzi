import cv2
import numpy as np

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    # Görüntüyü yakala
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü HSV renk uzayına çevir
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renk aralığını tanımla
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Kırmızı renge göre maske oluştur
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Konturları bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Her bir kontur için koordinatları hesapla ve çiz
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Küçük nesneleri filtrele
            # Minimum çevreleyen daireyi hesapla
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))

            # Balonun çevresine daire çiz ve koordinatları göster
            cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
            cv2.putText(frame, f"Koordinatlar: {center}", (center[0] - 50, center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Sonuçları göster
    cv2.imshow("Orijinal Görüntü", frame)
    cv2.imshow("Kırmızı Balonlar", mask)

    # Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
