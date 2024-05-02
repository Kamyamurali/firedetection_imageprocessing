import cv2
import numpy as np

fire_cascade = cv2.CascadeClassifier("fire_detection.xml")

cap = cv2.VideoCapture(0)
while True:
ret, frame = cap.read()
if not ret:
print("Error: Unable to capture frame")
break

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

fires = fire_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
if len(fires) > 0:
status_text = "Fire Detected"
text_color = (0, 255, 0)
for (x, y, w, h) in fires:
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
else:
status_text = "Not Detected"
text_color = (0, 0, 255)

cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

cv2.imshow("Fire Detection", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
break

cap.release()
cv2.destroyAllWindows()