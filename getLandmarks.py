import os
from mtcnn import MTCNN
import argparse
from PIL import Image
import cv2

def get_data_path(root='datasets/own_face'):
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith("png") or i.endswith("jpg")]
    return im_path

def detect_landmarks(im_path, root):
    detections = 'detections'
    if not os.path.exists(os.path.join(root, detections)):
        os.mkdir(os.path.join(root, detections))
    
    #initialize detector
    detector = MTCNN()
    
    for i in range(len(im_path)):
        im_name = im_path[i].split(os.path.sep)[-1].replace('.jpg', '').replace('.png', '')
        img = cv2.cvtColor(cv2.imread(im_path[i]), cv2.COLOR_BGR2RGB)
        face = detector.detect_faces(img)[0]
        txtfile = im_name + '.txt'
        txtpath = os.path.join(root, detections, txtfile)
        f = open(txtpath, 'w')
        for name, kp in face['keypoints'].items():
            f.write(str(kp[0]) + ' ' + str(kp[1]) + '\n')
        f.close 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_folder')
    args = parser.parse_args()
    root = args.img_folder
    detect_landmarks(get_data_path(root), root)





    
