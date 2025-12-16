import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Не удалось открыть камеру")
    exit()

# Фиксируем параметры СРАЗУ
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

time.sleep(0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Нет кадра")
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("CAMO TEST (DSHOW)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
