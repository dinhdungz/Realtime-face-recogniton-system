import os
import datetime
import torch
import tensorflow as tf
import csv
import cv2
import numpy as np
from scipy.special import softmax


def add_new_person(name, mode, detector, feature_extractor, save_path, path_image = None):
    '''
    This function adds a new person to the system. It takes in the following parameters: 
    
    # name (string): the name of the person being added 
    # mode (int): 0 for image, 1 for video 
    # detector (function): a model that detects faces in an image or video frame 
    # feature_extractor (function): a model that extracts features from an image of a face 
    # save_path (string): the path to save the image of the face 
    # path_image (string, optional): path to an image if mode is 0. If mode is 1, this parameter is not used. 

    # The function first reads in an image or video frame depending on the value of mode
    If mode is 0, it reads in an image from path_image
    If mode is 1, it captures frames from a video stream until 's' is pressed
    Then it uses detector to detect faces in the frame and extracts features from the detected face using feature_extractor
    It then saves the detected face as an image at save_path/name.jpg and writes both the feature vector and name to files.
    '''
    img = None
    if mode == 0 :
        img = cv2.imread(path_image)
    else:
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            cv2.imshow('Enter s to save image', img)
            if cv2.waitKey(1) == ord('s'):
                
                break
    
    detection = detector(img)
    result = detection.pandas().xyxy[0].to_numpy()
    if len(result) >= 1:
        i = result[0]
        x_min = int(i[0])
        x_max = int(i[2])
        y_min = int(i[1])
        y_max = int(i[3])
        img = img[y_min:y_max, x_min:x_max]
        img = cv2.resize(img, (125, 150))
        cv2.imwrite(f'{save_path}/{name}.jpg', img)
        img = np.expand_dims(img, axis = 0)
    else:
        print('Please choose other image')
        return 
    
    feature = feature_extractor.predict(img)
    np.set_printoptions(precision=4,formatter={'float_kind':'{:4f}'.format})

    write_feature(feature)
    write_name(name)

def predict(feature, all_feature, label):
    '''
    This function takes in three parameters: feature, all_feature, and label. 
    
    # feature is a feature vector of a single data point
    # all_feature is a matrix of all the features for all data points
    # label is an array of labels for each data point

    # The function calculates the Euclidean distance between the feature vector and all other feature vectors in the matrix
    It then uses the softmax function to calculate probabilities for each distance, and finds the index of the highest probability
    It then rounds this probability to two decimal places and returns it as 'acc'
    The function returns a tuple containing 'name' and 'acc'.
    '''
    distances = np.sum(np.square(feature - all_feature), axis = 1)
    prob = 1 - softmax(distances)
    index = np.argmax(prob)

    acc = np.around(np.max(prob), decimals = 2)
    name = label[index]
    return name, acc
    
def get_label(file):
    '''
    This function takes a file as an argument and returns a list of labels
    It opens the file in read mode, reads the lines from the file
    removes the newline character from each line and stores it in a list
    Finally, it closes the file and returns the list of labels.
    '''
    f = open(file, 'r')
    label = f.readlines()
    label = [name[:-1] for name in label]
    f.close()
    return label

def get_feature_array(file):
    '''
    This function takes a file as an argument and returns an array of the data from the file
    '''
    array = np.loadtxt(file, delimiter =',')
    return array

def write_name(name):
    '''
    This function write new person's name to file
    '''
    f = open('./Person/name.txt', 'a')
    f.write(f'{name}')
    f.write('\n')
    f.close()

def write_feature(feature):
    '''
    This function write new person's feature to file
    '''
    f = open('./Person/feature.csv', 'a')
    writer = csv.writer(f)
    writer.writerow(*feature)
    f.close()

def get_path():
    '''
    This function gets the path to the Attendance folder for the current date
    It checks if a folder for the current date exists in the Attendance folder
    and if not, it creates a folder with that name, creates an Attended_images folder inside it
    and creates a csv file with a header row called attended_table.csv
    Finally, it returns the path to the new or existing folder.
    '''
    list_atd = os.listdir('./Attendance')
    today = datetime.datetime.now()
    today = today.strftime('%d%m%y')
    if today in list_atd:
        pass
    else:
        header = ['Number', 'Name', 'Time']
        os.mkdir(f'./Attendance/{today}')
        os.chdir(f'./Attendance/{today}')
        os.mkdir('Attended_images')
        f = open('attended_table.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(header)
        f.close()
    path = f'./Attendance/{today}'
    return path

def create_attended_count(label):
    '''
    This function create dictionary with key = name in label and value = 0
    '''
    atd_count = {}
    for name in label:
        atd_count[name] = 0 
    return atd_count

def update(file, name, number):
    '''
    This function write infomation after attended success
    '''
    f = open(file, 'a')
    writer = csv.writer(f)
    time = datetime.datetime.now()
    time = time.strftime('%H:%M')
    writer.writerow([number, name, time])
    f.close()

def get_number(file):
    '''
    This function return number in attended file 
    '''
    f = open(file, 'r')
    reader = csv.reader(f)
    number = 0
    for line in reader:
        number +=1
    f.close()
    return number