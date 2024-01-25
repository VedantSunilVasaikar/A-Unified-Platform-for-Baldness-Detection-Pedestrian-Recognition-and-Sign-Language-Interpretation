import cv2
import mediapipe as mp
import csv
import copy
import itertools
import string
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(letter, landmark_list):
    csv_path = 'keypoint.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([letter, *landmark_list])


alphabet = list(string.ascii_uppercase)
alphabet +=  ['1','2','3','4','5','6','7','8','9']

address = 'images/data/'
IMAGE_FILES = []
for i in alphabet:
  for j in range(1199):
    IMAGE_FILES.append(address+i+'/'+str(j)+'.jpg')
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    
    image = cv2.flip(cv2.imread(file), 1)
    
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
        landmark_list = calc_landmark_list(annotated_image, hand_landmarks)
        pre_processed_landmark_list = pre_process_landmark(landmark_list)
        logging_csv(file[12],pre_processed_landmark_list)