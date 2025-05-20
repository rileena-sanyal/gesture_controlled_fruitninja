import cv2
import mediapipe as mp
import random
from ultralytics import YOLO


"""
Created by Rileena.

Please note: this project was created for personal entertainment and is
not maintained. It does not have elaborate commenting or docstrings. 

Email: rileena08@gmail.com

"""
class FruitNinjaGame:
    def __init__(self):
        """
        Initialisation.

        This is to set up all required global variables
        and to initialise YOLO for object detection and
        MediaPipe for hand pose estimation.

        Please note, MediaPipe is enough for this project,
        YOLO has been added for aesthetic reasons.
        """
        self.yolo_model = YOLO('yolo11l.pt')
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.score = 0  # to keep score
        self.missed = 0  # number of fruits missed
        self.fruits = []  # list to store fruits
        self.max_fruits = 5  # maximum no. of fruits on the screen at a time
        self.max_misses = 3  # maximum no. of consecutive misses allowed
        self.win_score = 15  # winning score

        self.cap = cv2.VideoCapture(1)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.miss_line_y = 500

        self.left_hand_track_points = []
        self.right_hand_track_points = []
        self.frame_counter = 0
        self.current_fruit_count = 1
        self.generate_initial_fruits()

    def generate_fruit(self):
        """
        Generates a random fruit with a random position, speed, and color.

        Returns:
            dict: A dictionary representing a fruit with attributes like
                  position (x, y), speed, color, hit status, and slash details.
        """
        colors = [(0, 0, 255), (0, 255, 0),
                  (255, 0, 0), (0, 255, 255),
                  (255, 0, 255)]
        return {
            'x': random.randint(50, self.width - 50),
            'y': 50,
            'speed': random.randint(5, 10),
            'color': random.choice(colors),
            'hit': False,
            'slash': None
        }

    def generate_initial_fruits(self):
        """
        Generates an initial set of fruits for the game.

        The number of fruits generated is based on the current fruit count.
        """
        for _ in range(self.current_fruit_count):
            self.fruits.append(self.generate_fruit())

    def process_frame(self):
        """
        Processes a single frame of the game, including detecting hands,
        tracking fruits, and updating the score and missed count.

        Args:
            None

        Returns:
            tuple: A tuple containing a boolean indicating whether the game
                   should continue and the current frame.
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        cv2.line(frame, (0, 500), (800, 500), (255, 255, 255), 2)
        results = self.yolo_model.track(frame, tracker='botsort.yaml')
        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x, index_y = int(index_finger.x * self.width), int(index_finger.y * self.height)

                if hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x < 0.5:
                    self.left_hand_track_points.append((index_x, index_y))
                else:
                    self.right_hand_track_points.append((index_x, index_y))

                for fruit in self.fruits:
                    if fruit['x'] - 30 < index_x < fruit['x'] + 30 and fruit['y'] - 30 < index_y < fruit['y'] + 30:
                        fruit['hit'] = True
                        self.score += 1
                        fruit['slash'] = ((fruit['x'] - 20, fruit['y'] - 20), (fruit['x'] + 20, fruit['y'] + 20))

        self.frame_counter += 1
        if self.frame_counter >= 30:
            self.left_hand_track_points.clear()
            self.right_hand_track_points.clear()
            self.frame_counter = 0

        for i in range(1, len(self.left_hand_track_points)):
            cv2.line(frame, self.left_hand_track_points[i - 1], self.left_hand_track_points[i], (0, 255, 255), 2)
        for i in range(1, len(self.right_hand_track_points)):
            cv2.line(frame, self.right_hand_track_points[i - 1], self.right_hand_track_points[i], (255, 0, 255), 2)

        new_fruits = []
        for fruit in self.fruits:
            if not fruit['hit']:
                cv2.circle(frame, (fruit['x'], fruit['y']), 20, fruit['color'], -1)
                fruit['y'] += fruit['speed']
                if fruit['y'] > self.miss_line_y:
                    self.missed += 1
                else:
                    new_fruits.append(fruit)
            else:
                if fruit['slash']:
                    cv2.line(frame, fruit['slash'][0], fruit['slash'][1], (255, 0, 0), 3)
        self.fruits = new_fruits

        if len(self.fruits) == 0:
            self.current_fruit_count = min(self.current_fruit_count + 1, self.max_fruits)
        while len(self.fruits) < self.current_fruit_count:
            self.fruits.append(self.generate_fruit())

        cv2.putText(frame, f'Score: {self.score}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        cv2.putText(frame, f'Missed: {self.missed}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

      # Check win/lose conditions
        if self.missed >= self.max_misses:
            for _ in range(80):
                cv2.putText(frame, 'Game Over!', (100, 495),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                cv2.imshow('Fruit Ninja', frame)
                cv2.waitKey(80)
            return False, frame

        if self.score >= self.win_score:
            for _ in range(80):
                cv2.putText(frame, 'You Win!', (100, 495),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow('Fruit Ninja', frame)
                cv2.waitKey(180)
            return False, frame

        cv2.imshow('Fruit Ninja', frame)
        return True, frame

    def run(self):
        """
        Starts the main game loop, continuously processing frames and updating the game state.

        The loop ends when the game is over or the user presses the 'q' key.
        """
        while self.cap.isOpened():
            success, frame = self.process_frame()
            if not success or cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = FruitNinjaGame()
    game.run()
