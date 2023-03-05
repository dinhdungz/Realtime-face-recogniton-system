from Realtime_face_recognition.utils_function import *

detector = torch.hub.load('ultralytics/yolov5', 'custom', path = './Model/face_detection_yolov5s.pt')
feature_extractor = tf.keras.models.load_model('./Model/feature_extractor')
cap = cv2.VideoCapture(0) #edit

path = get_path()
label = get_label('./Person/name.txt')
attended_count = create_attended_count(label)

while True:
    ret, image = cap.read()
    detection = detector(image)
    results = detection.pandas().xyxy[0].to_numpy()
    for i in results:
        if i[4] >= 0.6:
            x_min = int(i[0])
            x_max = int(i[2])
            y_min = int(i[1])
            y_max = int(i[3])
            img = image[y_min:y_max, x_min:x_max]
            img = cv2.resize(img, (125, 150))
            img_exp = np.expand_dims(img, axis = 0)
            feature = feature_extractor.predict(img_exp)

            label = get_label('./Person/name.txt')
            feature_array = get_feature_array('./Person/feature.csv')
            name, acc = predict(feature, feature_array, label)
            label = name
            # process
            if acc > 0.85:
                attended_count[name] +=1
                if attended_count[name] > 20:
                    
                    label = 'Completed Attendance'
                image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                image = cv2.putText(image, f'{label}', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            else:
                label = 'Unknown'
                image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                image = cv2.putText(image, f'{label}', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            

            if attended_count[name] == 20:
                number = get_number(f'{path}/attended_table.csv')
                update(f'{path}/attended_table.csv', name, number)
                cv2.imwrite(f'{path}/Attended_images/{name}.jpg',img)
            
        else:
            continue
    cv2.imshow('Recognition - Enter q to exit', image)
        
    if cv2.waitKey(1) == ord('q'):
        break
