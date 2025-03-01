import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_fingers(hand_landmarks):
    fingers = 0

    if hand_landmarks[17].x < hand_landmarks[5].x:  
     if hand_landmarks[4].x > hand_landmarks[3].x:  
        fingers += 1
    else:  
      if hand_landmarks[4].x < hand_landmarks[3].x:
        fingers += 1

    for tip_id in [8, 12, 16, 20]:  
        if hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y:
            fingers += 1
    return fingers

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers_count = count_fingers(hand_landmarks.landmark)
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.putText(frame, f'Fingers: {fingers_count}', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
