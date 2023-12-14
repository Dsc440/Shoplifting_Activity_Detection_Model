from keras.models import load_model
import numpy as np
import cv2
image_size=(224,224)
loaded_model=load_model('/kaggle/input/modelh5/my_model.h5')
video_path="/kaggle/input/shoplifting-dataset/Shoplifting dataset/Shoplifting/Shoplifting (16).mp4"
count = 0
video=[]
video_reader = cv2.VideoCapture(video_path)
n=int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video_reader.get(3))
frame_height = int(video_reader.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'),30, size)
if n<=150:
    print("The length of input video is very less.")
while True:
    is_true, frame = video_reader.read()
    write_frame = frame
    if len(video) >= 150:
        break
    if is_true:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
        video.append(np.array(frame))
        result.write(write_frame)
        count += 1
    else:
        break
video = np.array(video)
while True:
    is_true, frame =video_reader.read()
    write_frame = frame
    count+=1
    print(f"frame {count} of {n}")
    if is_true:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
        newvideo=[]
        for i in range(1,len(video)):
            newvideo.append(video[i])
        newvideo.append(frame)
        video=np.array(newvideo)
        video=np.expand_dims(video,axis=0)
        shoplifting = loaded_model.predict(video)
        normal = "{:.2f}".format(100-shoplifting[0][0]*100)
        shoplifting = "{:.2f}".format(shoplifting[0][0]*100)
        video=video[0]
        text = f"Normal:{normal}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_position = (10, 40)
        cv2.putText(write_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 2)
        text_position = (10, 80)
        cv2.putText(write_frame,f"Shoplifting:{shoplifting}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 2)
        result.write(write_frame)
    else:
        break
video_reader.release()
result.release()