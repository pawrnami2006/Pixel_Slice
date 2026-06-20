import cv2
import mediapipe as mp
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
trail_points = []

# Open webcam
cap = cv2.VideoCapture(0)

apple_img = cv2.imread(
    "assets/apple.png",
    cv2.IMREAD_UNCHANGED
)

apple_img = cv2.resize(apple_img, (80, 80))

apple_x = 300
apple_y = 700

apple_vx = 3
apple_vy = -20

gravity = 0.5

def overlay_png(background, overlay, x, y):

    h, w = overlay.shape[:2]

    if x < 0 or y < 0:
        return

    if x + w > background.shape[1]:
        return

    if y + h > background.shape[0]:
        return

    alpha = overlay[:, :, 3] / 255.0

    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c]
            +
            (1 - alpha)
            * background[y:y+h, x:x+w, c]
        )

while True:
    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    apple_vy += gravity

    apple_x += apple_vx
    apple_y += apple_vy

    # Respawn apple
    if apple_y > frame.shape[0]:

        apple_x = random.randint(100, 500)

        apple_y = frame.shape[0]

        apple_vx = random.choice([-4, -3, 3, 4])

        apple_vy = random.randint(-22, -18)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Get index finger tip
            h, w, c = frame.shape

            index_tip = hand_landmarks.landmark[8]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            # Draw green circle on fingertip
            cv2.circle(frame, (x, y), 15, (0, 255, 0), -1)

            trail_points.append((x, y))

            if len(trail_points) > 20:
                trail_points.pop(0)
                
    # Draw trail
    for i in range(1, len(trail_points)):
        cv2.line(frame, trail_points[i-1], trail_points[i], (255, 255, 255), 5)

        overlay_png(
            frame,
            apple_img,
            int(apple_x),
            int(apple_y)
        )

    cv2.imshow("Pixel Slice", frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()