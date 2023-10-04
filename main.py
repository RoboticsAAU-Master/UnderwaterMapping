# Standard imports


# Own imports
from UW_VO import load


if __name__ == "__main__":
    cap = load.load_monocular(r"Datasets\Kridtgraven\Normal", imu=False)
    
    #load.play_video(r"datasets\Kridtgraven\Normal.mp4", scale=0.5)
    