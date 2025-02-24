import cv2
import numpy as np

def count_fingers(thresh_img, frame):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:

        max_contour = max(contours, key=cv2.contourArea)

        hull = cv2.convexHull(max_contour, returnPoints=False)

        defects = cv2.convexityDefects(max_contour, hull)

        count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                a = np.linalg.norm(np.array(start) - np.array(end))
                b = np.linalg.norm(np.array(start) - np.array(far))
                c = np.linalg.norm(np.array(end) - np.array(far))

                angle = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2 * b * c)))

                if angle < 90:
                    count += 1
                    cv2.circle(frame, far, 5, (0, 255, 0), -1)  

        cv2.putText(frame, f'Fingers: {count+1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    
    count_fingers(mask, frame)

    cv2.imshow("Hand Detection", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()