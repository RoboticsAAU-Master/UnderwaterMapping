import cv2 as cv
import numpy as np

def play_video(path : str, scale : float = 1) -> None:
    """Function to play back video

    Args:
        path (str): Relative path  
        scale (list): To scale the image
        
    Returns:
        None
    """
    cap = cv.VideoCapture(path)

    if not cap.isOpened():
        raise Exception("Cannot Open Camera")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        frame = cv.resize(frame, (0,0), fx=scale, fy=scale)
        
        #Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

