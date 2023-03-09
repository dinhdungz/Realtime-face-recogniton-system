from util import *

detector = torch.hub.load('ultralytics/yolov5', 'custom', path = './Model/face_detection_yolov5s.pt')
feature_extractor = tf.keras.models.load_model('./Model/feature_extractor')

while True:
    print('\n')
    print('0. End \n')
    print('1. Add new person \n')
    print('2. Show attended \n')
    print('3. Attendance \n')


    select = input('Enter selection from (0-3): ')
    if select == '0':
        break
    elif select == '1':
        name = input('Enter name: ')
        mode = input('Enter 0 (camera) - file_video: ')
        add_new_person(name, detector, feature_extractor, mode)
    elif select == '2':
        path = get_path()
        atd_path = os.path.join(path, 'attended_table.csv')
        show_atd(atd_path)
    elif select == '3':
        recognition(detector, feature_extractor)
    else:
        print('Please selection from (0-3)')