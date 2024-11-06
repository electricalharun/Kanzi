import cv2
import numpy as np

# Yüz tespiti için yüz tanıma modeli yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Küçük kırmızı balon için maksimum alan değeri (deneme yaparak ayarlayabilirsiniz)
max_area_for_small_balloon = 1000  # Küçük balonun maksimum alanı

while True:
    # Görüntüyü yakala
    ret, frame = cap.read()
    if not ret:
        break

    # Griye dönüştürerek yüz algıla
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Yüz algılandığında balon algılamaya geç
    if len(faces) > 0:
        # HSV renk uzayına çevir
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Kırmızı renk aralığı tanımla
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

        # Küçük kırmızı balonları filtrele ve göster
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max_area_for_small_balloon:  # Sadece küçük balonları seç
                # Minimum çevreleyen daireyi hesapla
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))

                # Küçük kırmızı balonun çevresine daire çiz
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)

    # Sonuçları göster
    cv2.imshow("Yüz ve Kırmızı Balon Algılama", frame)

    # Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()