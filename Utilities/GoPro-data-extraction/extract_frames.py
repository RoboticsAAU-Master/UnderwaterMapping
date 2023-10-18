import cv2
import os
from tqdm import tqdm


def save_all_frames(video_path, dir_path, basename, ext, frame_keep):
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    
    n = 0
    for f in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if ret:
            if (n % frame_keep) == 0:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame_gray)
            n += 1
            
        else:
            return
        

if __name__ == "__main__":
    # Input: path_to_video  output_folder  base_image_name  extension
    save_all_frames('C3_GX040003.MP4', 
                    'Utilities/Bag-conversion/Output/C3_GX040003/Images', 
                    'sample_video_img', 
                    'png', 
                    1)
