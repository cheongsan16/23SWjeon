import cv2
import numpy as np
from tensorflow.keras.models import load_model

model_gender = load_model("C:\\Users\\82104\\PycharmProjects\\pythonProject\\_gender_model.h5")
model_age = load_model("C:\\Users\\82104\\PycharmProjects\\pythonProject\\_age_model_1.h5")

input_size = (200, 200, 3)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 성별과 나이 클래스
gender_list = ['Male', 'Female']
age_list = ['(0 ~ 2)','(3 ~ 5)','(6 ~ 12)','(13 ~ 18)', '(19 ~ 24)','(25 ~ 35)','(36 ~ 45)','(46 ~ 60)','(61 ~ 80)','(81 ~ 120)']

video_capture = cv2.VideoCapture(0)

while True:
    # 영상 프레임 읽기
    ret, frame = video_capture.read()

    # 영상을 흑백으로 변환
    #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출 수행
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 얼굴에 사각형 그리기
    for (x, y, w, h) in faces:
        # 얼굴 영역 추출
        face_roi = frame[y:y + h, x:x + w]

        # 입력 크기에 맞게 조절
        face_roi = cv2.resize(face_roi, (input_size[1], input_size[0]))  # dsize 수정

        # 모델에 입력하기 위해 차원 확장
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)

        # 성별과 나이 예측
        gender_prob = model_gender.predict(face_roi)
        age_prob = model_age.predict(face_roi)
        gender_label = gender_list[np.argmax(gender_prob[0])]
        age_label = age_list[np.argmax(age_prob[0])]

        # 영상에 결과 표시
        label = f'{gender_label}, {age_label}'
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 화면에 영상 출력q
    cv2.imshow('Face Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
