import os
import cv2
import argparse


# TODO: fix offset issue (caused by syncing position, possibly also cv buffer size)
#test both 0 and end of vid functionality
"""
release all resources, and other cv2 objects
"""
def release(resources: list):
    for r in resources:
        r.release()
    cv2.destroyAllWindows()

"""
create a new recording of the video at path, clipped into foi[0] and foi[1]
"""
def trim(path, foi):
    curr, end = foi[0], foi[1]
    suffix = f'_trimmed_{curr},{end}.mp4'
    # [:-4] removes the .mp4 from file path
    target_path = path[:-4] + suffix
    # codes for compression method
    code = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
    fps = 30

    cap = cv2.VideoCapture(path)
    
    #CV2 FAILS SILENTLY A LOT. So, we must be diligent with error handling
    if not cap.isOpened():
        release([cap])
        raise SystemError("Error opening the file. Aborting.")
    
    vid_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if end > vid_length or curr < 1:
        release([cap])
        raise ValueError(f"Frame out of range for video. The video is of length {vid_length}. Aborting.")
   

    cap.set(cv2.CAP_PROP_POS_FRAMES, curr-1)
    succ, frame = cap.read()
    #I have absolutely no clue why the VideoWriter requires a transposed frame shape
    frame_shape = (frame.shape[1], frame.shape[0]) 
    vid = cv2.VideoWriter(target_path, code, 30, frame_shape)
    while curr < end:
        if not succ:
            release([cap,vid])
            raise SystemError("Error reading from file at frame {curr}. Aborting.")
        else:
            vid.write(frame)
        curr += 1
        succ, frame = cap.read()
    if succ:
        # don't save if the loop was broken out of
        print(f'Video saved to {target_path}')
    release([cap, vid])

"""
visualize the image at poition frame in the video given by path. Assumes that path is .mp4 file]

"""

def frame(path, frame_num):
    suffix = f'_frame{frame_num}.png'
    # [:-4] removes the .mp4 from file path
    target_path = path[:-4] + suffix
    cap = cv2.VideoCapture(path)
    
    #CV2 FAILS SILENTLY A LOT. So, we must be diligent with error handling
    if not cap.isOpened():
        release([cap])
        raise SystemError("Error opening the file. Aborting.")
    
    vid_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_num > vid_length or frame_num < 1:
        release([cap])
        raise ValueError(f"Frame out of range for video. The video is of length {vid_length}. Aborting.")
    else:
        # set to one before frame of interest so that read() returns correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
        succ, frame = cap.read()
        if not succ:
            release([cap])
            raise SystemError('Frame could not be read. Aborting.')
        elif cv2.imwrite(target_path, frame):
            print("Frame %d saved at path %s" % (frame_num, target_path))
        else:
            raise SystemError ("Error saving the file")
    
    release([cap])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Video trimmings')
    parser.add_argument('-foi', '--frames_of_interest',
                        help='Frames of interest, comma separated. If one number, a still image is generated. \
                          If two numbers are given, the recording is trimmed to a frame range ')
    parser.add_argument('-p', '--path', help='path to .mp4 video file')

    args = parser.parse_args()

    foi_str = args.frames_of_interest.split(',')
    foi = [int(s) for s in foi_str]
    if not os.path.exists(args.path):
        raise FileNotFoundError(
            'Could not find the video at the path provided')
    elif len(foi) != 1 and len(foi) != 2:
        raise ValueError(
            'Only inputs of one or two frames are currently supported')
    elif len(foi) == 1:
        frame(args.path, foi[0])
    elif len(foi) == 2:
        trim(args.path, foi)
