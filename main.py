import pathlib
import sys
import time
import csv
from draw_utils import *
from facemesh import *
from kalman import *
import numpy as np
import tensorflow as tf


# 모델 경로 세팅
ENABLE_EDGETPU = False

MODEL_PATH = pathlib.Path("./models/")
if ENABLE_EDGETPU:
    DETECT_MODEL = "cocompile/face_detection_front_128_full_integer_quant_edgetpu.tflite"
    MESH_MODEL = "cocompile/face_landmark_192_full_integer_quant_edgetpu.tflite"
    PRED_MODEL = "speak_predict3.tflite"
else:
    DETECT_MODEL = "face_detection_front.tflite"
    MESH_MODEL = "face_landmark.tflite"
    PRED_MODEL = "speak_predict3.tflite"


# 카메라 초기 세팅
cap = cv2.VideoCapture(-1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
ret, init_image = cap.read()
if not ret:
    sys.exit(-1)

# Face Models 초기화
face_detector = FaceDetector(model_path=str(MODEL_PATH / DETECT_MODEL), edgetpu=ENABLE_EDGETPU)
face_mesher = FaceMesher(model_path=str((MODEL_PATH / MESH_MODEL)), edgetpu=ENABLE_EDGETPU)
face_aligner = FaceAligner(desiredLeftEye=(0.38, 0.38))
face_pose_decoder = FacePoseDecoder(init_image.shape)

# Prediction Model 초기화
predict_interpreter = tf.lite.Interpreter(str(MODEL_PATH / PRED_MODEL))
predict_interpreter.allocate_tensors()
input_details = predict_interpreter.get_input_details()
output_details = predict_interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Introduce scalar stabilizers for pose.
pose_stabilizers = [Stabilizer(
    initial_state=[0, 0, 0, 0],
    input_dim=2,
    cov_process=0.2,
    cov_measure=2) for _ in range(6)]





# detect single frame
def detect_single(image):
    # pad image
    h, w, _ = image.shape
    target_dim = max(w, h)
    padded_size = [(target_dim - h) // 2,
                   (target_dim - h + 1) // 2,
                   (target_dim - w) // 2,
                   (target_dim - w + 1) // 2]
    # padding needed for the top, bottom, left, and right sides of the image to make it square. 
    padded = cv2.copyMakeBorder(image.copy(),
                                *padded_size,
                                cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
    # used to pad the image with zeros (black color) based on the calculated padding sizes.
    padded = cv2.flip(padded, 3) 
    # 3 argument in cv2.flip indicates horizontal flipping. This operation is performed to augment the data, providing a mirrored version of the image.

    # face detection
    bboxes_decoded, landmarks, scores = face_detector.inference(padded)

    mesh_landmarks_inverse = []
    r_vecs, t_vecs = [], []
    for i, (bbox, landmark) in enumerate(zip(bboxes_decoded, landmarks)):
        # landmark detection
        aligned_face, M = face_aligner.align(padded, landmark)
        mesh_landmark, _ = face_mesher.inference(aligned_face)
        mesh_landmark_inverse = face_aligner.inverse(mesh_landmark, M)
        mesh_landmarks_inverse.append(mesh_landmark_inverse)

        # pose detection
        r_vec, t_vec = face_pose_decoder.solve(landmark)
        r_vecs.append(r_vec)
        t_vecs.append(t_vec)

        # tracking
        if i == 0:
            landmark_stable = []
            for mark, stb in zip(landmark.reshape(-1, 2), pose_stabilizers):
                stb.update(mark)
                landmark_stable.append(stb.get_results())
            landmark_stable = np.array(landmark_stable).flatten()
            landmarks[0] = landmark_stable

    # draw
    image_show = draw_face(padded, bboxes_decoded, landmarks, scores, confidence=True)
    lip_coords = []
    for i, mesh_landmark_inverse in enumerate(mesh_landmarks_inverse):
        lip_coords.append(np.array(mesh_landmark_inverse[[0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415], :]))
        
       

        image_show = draw_mesh(image_show, mesh_landmark_inverse, contour=True)

    image_show = image_show[padded_size[0]:target_dim - padded_size[1], padded_size[2]:target_dim - padded_size[3]]
    return image_show, lip_coords




##############################################################################################################
# max_rows = 2000
# rows_written = 0
pred_frames = np.zeros((19, 120))
recent_frame = np.zeros((1,120))
target_fps = 20  
prediction = ""


while True:
    start_time = time.time() # 시작 시간

    ret, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Frame detection
    image_show, lip_coords = detect_single(image)

    # 버그 수정: 얼굴이 인식 안되었을때 처리 코드
    if lip_coords == []:
        continue

    result = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)

    # 최근 20개 Frame을 활용한 현재 Frame의 발화 여부 예측
    # current_frame = lip_coords[0].reshape(1,120)
    # new_pred_frame = np.abs(current_frame - recent_frame)
    # pred_frames = np.vstack((pred_frames[1:], new_pred_frame))
    # recent_frame = current_frame 
    # input_data = np.array(pred_frames, dtype=np.float64)
    # predict_interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_data, axis=0))
    # predict_interpreter.invoke()
    # output_data = predict_interpreter.get_tensor(output_details[0]['index'])
    # if output_data[0][0] > 0.5:
    #     prediction = 'SPEAKING'
    # else:
    #     prediction='NOT SPEAKING'
    

    # print(output_data)


    # 고정된 fps를 위한 delay 세팅
    end_time = time.time() # 끝 시간
    elapsed_time = end_time - start_time
    delay_time = max(0, 1 / target_fps - elapsed_time) # 고정된 fps를 위한 delay시간
    time.sleep(delay_time)
    end_time2 = time.time()
    fps = 1 / (end_time2 - start_time) # 해당 루프에서의 fps

    # 화면 Display 코드
    cv2.putText(result, 'FPS:%5.2f'%(fps), (10,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,  color = (0,255,0), thickness = 1)
    # cv2.putText(result, 'Prediction:%s %.5f'%(prediction, output_data[0][0]), (30,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,  color = (0,0,255), thickness = 1)

    cv2.imshow('demo', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ### Reference: 데이터 수집을 위해 활용했었던 코드

    # if rows_written < max_rows:
    #     with open('output3.csv', 'a', newline='') as csvfile:
    #         csv_writer = csv.writer(csvfile)
    #         # Write the list as a row in the CSV file
    #         for lip_coord in lip_coords:
    #             csv_writer.writerow(lip_coord)
    #     rows_written += 1