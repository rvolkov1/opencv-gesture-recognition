import csv
import os
import mediapipe as mp

mp_hands  = mp.solutions.hands

class Data():
    def __init__(self):
        self.collection_on = False
        self.file = '../data/landmarks.csv'

    def delete_data_file(self):
        os.remove(self.file)

    def toggle_collection(self):
        self.collection_on = not self.collection_on
    
    def add_data_point(self, gesture, landmarks):
        if (not self.collection_on): return

        with open(self.file, 'a', newline='\n') as data:
            writer = csv.writer(data, delimiter = ',')

            landmarks.insert(0, gesture)

            writer.writerow(landmarks)

    def load_data(self):
        x = []
        y = []
        
        with open(self.file, newline='\n') as file:
            reader = csv.reader(file, delimiter=",")

            for row in reader:
                points = []
                for i, value in enumerate(row):
                    if i == 0:
                        y.append(int(value))
                    else:
                        points.append(float(value))
                x.append(points)

            return x, y

    @staticmethod
    def landmarks_to_list(landmarks):
        new_list = []
        for hand_landmarks in landmarks:  
            for i in range(21):    
                new_list.append(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x)
                new_list.append(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y)
                new_list.append(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z)
                    
        return new_list

