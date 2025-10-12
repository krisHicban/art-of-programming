import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

results = hands.process(frame)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Get thumb tip and index tip
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]

        # Calculate pinch distance
        distance = np.sqrt(
            (thumb.x - index.x)**2 +
            (thumb.y - index.y)**2
        )

        if distance < 0.05:
            print("Pinch detected! 📸")