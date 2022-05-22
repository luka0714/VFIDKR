import numpy as np
from PIL import Image
import os
import cv2

# ori_filepath = "E:/UCF101/UCF-101/"
# final_filepath = "C:/Users/Nemo/Desktop/triplets_UCF/"

 
video_src_src_path = 'E:/UCF101/UCF-101/' #数据集路径
video_save_save_path = 'C:/Users/Nemo/Desktop/triplets_UCF/'
label_name = os.listdir(video_src_src_path)
# label_dir = {}
# index = 0
count = 0
for type in label_name:

    video_list = os.path.join(video_src_src_path, type)
    video_list += '/'

    for video_name in os.listdir(video_list):
        count += 1
        if count % 40 == 1:

            if not os.path.exists(video_save_save_path + str(count)):
                os.mkdir(video_save_save_path + str(count))
            
            video_src_path = os.path.join(video_list, video_name)

            each_video_save_full_path = os.path.join(video_save_save_path, str(count)) + '/'

            cap = cv2.VideoCapture(video_src_path)
            frame_count = 0
            success = True
            while success:
                frame_count += 1
                success, frame = cap.read()

                if frame_count == 2 or frame_count == 3 or frame_count == 4:

                    params = []
                    params.append(1)
                    if success:
                        cv2.imwrite(each_video_save_full_path + "im%d.png" % frame_count, frame, params)
                
            cap.release()


    # if i.startswith('.'):
    #     continue

    # label_dir[i] = index
    # index += 1
    # video_src_path = os.path.join(video_src_src_path, i)

    # video_save_path = os.path.join(video_save_save_path, i) + '_png'
    # if not os.path.exists(video_save_path):
    #     os.mkdir(video_save_path)
 
    # videos = os.listdir(video_save_save_path)
    # 过滤出avi文件
    # videos = filter(lambda x: x.endswith('avi'), videos)
 
    # for each_video in video_src_path:
    # each_video_name, _ = i.split('.')
    # if not os.path.exists(video_save_save_path + each_video_name):
    #     os.mkdir(video_save_save_path + each_video_name)

    # each_video_save_full_path = os.path.join(video_save_save_path, each_video_name) + '/'

    # each_video_full_path = video_src_path

    # cap = cv2.VideoCapture(each_video_full_path)
    # frame_count = 1
    # success = True
    # while success:
    #     success, frame = cap.read()
    #     # print('read a new frame:', success)

    #     params = []
    #     params.append(1)
    #     if success:
    #         cv2.imwrite(each_video_save_full_path + "%03d.png" % frame_count, frame, params)

    #     frame_count += 1
    # cap.release()
# np.save('label_dir.npy', label_dir)
# print(label_dir)
