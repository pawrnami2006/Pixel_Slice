import cv2
import mediapipe as mp
import random
import time

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

success, frame = cap.read()

screen_height = frame.shape[0]

apple_img = cv2.imread(
    "assets/apple.png",
    cv2.IMREAD_UNCHANGED
)

orange_img = cv2.imread(
    "assets/orange.png",
    cv2.IMREAD_UNCHANGED
)

strawberry_img = cv2.imread(
    "assets/strawberry.png",
    cv2.IMREAD_UNCHANGED
)

watermelon_img = cv2.imread(
    "assets/watermelon.png",
    cv2.IMREAD_UNCHANGED
)

game_over_img = cv2.imread(
    "assets/game_over.png",
    cv2.IMREAD_UNCHANGED
)

title_img = cv2.imread(
    "assets/title.png",
    cv2.IMREAD_UNCHANGED
)

heart_img = cv2.imread(
    "assets/heart.png",
    cv2.IMREAD_UNCHANGED
)

heart_img = cv2.resize(
    heart_img,
    (60, 60)
)

game_over_img = cv2.resize(
    game_over_img,
    (500, 250)
)

title_img = cv2.resize(
    title_img,
    (500, 250)
)

apple_img = cv2.resize(apple_img, (80, 80))
orange_img = cv2.resize(orange_img, (80, 80))
strawberry_img = cv2.resize(strawberry_img, (80, 80))
watermelon_img = cv2.resize(watermelon_img, (80, 80))

fruit_images = [
    apple_img,
    orange_img,
    strawberry_img,
    watermelon_img
]

apple_x = 300
apple_y = screen_height - 100

apple_vx = 3
apple_vy = -20

gravity = 0.5

apple_radius = 40

score = 0
lives = 3

game_started = False

current_fruit = random.choice(fruit_images)

last_hit_time = 0

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

    if not game_started:

        frame = cv2.flip(frame, 1)

        overlay_png(
            frame,
            title_img,
            70,
            100
        )

        cv2.putText(
            frame,
            "Press SPACE to Start",
            (140, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("Pixel Slice", frame)

        key = cv2.waitKey(1)

        if key == 32:
            game_started = True

        continue

    if lives <= 0:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            overlay_png(
                frame,
                game_over_img,
                70,
                100
            )

            cv2.putText(
                frame,
                f"Final Score: {score}",
                (180, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press ESC to Quit",
                (150, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("Pixel Slice", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

        break


    # Mirror effect
    frame = cv2.flip(frame, 1)

    apple_vy += gravity

    apple_x += apple_vx
    apple_y += apple_vy

    # Bounce from left wall
    if apple_x <= 0:
        apple_vx *= -1

    # Bounce from right wall
    if apple_x >= frame.shape[1] - 80:
        apple_vx *= -1

    # Bounce from top
    if apple_y <= 0:
        apple_vy = abs(apple_vy)

    # Respawn apple
    if apple_y > frame.shape[0]:

        lives -= 1

        apple_x = random.randint(
            50,
            frame.shape[1] - 100
        )

        apple_y = frame.shape[0]

        apple_vx = random.randint(-8, 8)

        apple_vy = random.randint(-28, -16)

        current_fruit = random.choice(fruit_images)

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

    for point in trail_points:

        trail_x, trail_y = point

        distance = (
            (trail_x - (apple_x + 40)) ** 2
            +
            (trail_y - (apple_y + 40)) ** 2
        ) ** 0.5

        if distance < apple_radius and time.time() - last_hit_time > 0.3:

            last_hit_time = time.time()

            score += 1

            apple_x = random.randint(
                50,
                frame.shape[1] - 100
            )

            apple_y = frame.shape[0]

            apple_vx = random.randint(-8, 8)

            apple_vy = random.randint(-28, -16)

            current_fruit = random.choice(fruit_images)
                
    # Draw trail
    for i in range(1, len(trail_points)):
        cv2.line(
            frame,
            trail_points[i-1],
            trail_points[i],
            (255, 255, 255),
            5
        )

    # Draw apple
    overlay_png(
        frame,
        current_fruit,
        int(apple_x),
        int(apple_y)
    )

    # Draw score
    cv2.putText(
        frame,
        f"Score: {score}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Draw lives
    for i in range(lives):

        overlay_png(
            frame,
            heart_img,
            20 + (i * 70),
            70
        )

    cv2.imshow("Pixel Slice", frame)

    # Press ESC to quit
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()