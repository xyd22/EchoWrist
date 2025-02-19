from data_collection import data_record
import argparse
import random
import os

def speed_study(camera, path_prefix, is_warmup, is_maincollect):
    assert not is_warmup==is_maincollect, "cannot warmup and collect at the same time"
   
    durations = [1.5, 2.0, 2.5]
    duration_reps = 7

    if not os.path.exists(path_prefix): # initialize the directory
        os.mkdir(path_prefix)
        os.mkdir(os.path.join(path_prefix, "test"))
        for duration in durations:
            os.mkdir(os.path.join(path_prefix, str(duration)))
        duration_orders = durations * duration_reps
        random.shuffle(duration_orders)
        print(duration_orders)
        order_file = open(os.path.join(path_prefix, "order.txt"), "w")
        for idx, duration in enumerate(duration_orders):
            order_file.write(str(duration))
            if idx < len(duration_orders)-1:
                order_file.write('\n')
        order_file.close()

    folds = 2 if is_maincollect else 1
    reps = 2
    is_noserial = True
    countdown = 12
    is_right = True
    is_audio = False
    folder_name = ""

    if is_warmup:
        curr_duration = 1.5
        next_duration = 0
        folder_name = "test"
    
    if is_maincollect:
        if not os.path.exists(os.path.join(path_prefix, "progress.txt")):
            curr_order = 0
        else:
            with open(os.path.join(path_prefix, "progress.txt"), 'r') as fp:
                lines = len(fp.readlines())
            curr_order = lines

        curr_duration = 0
        next_duration = 0
        with open(os.path.join(path_prefix, "order.txt")) as fp:
            for i, line in enumerate(fp):
                if i == curr_order:
                    curr_duration = float(line.strip())
                    folder_name = str(curr_duration)
                if i == curr_order+1:
                    next_duration = float(line.strip())
        assert not curr_duration == 0, "all durations collected already"
        print("CURRENT DURATION: ", curr_duration, ", session number: ", curr_order + 1)

    data_record(path_prefix, folder_name, "5fingers", curr_duration, folds, reps, is_noserial, countdown, camera, is_audio, is_right)


    if is_maincollect:
        with open(os.path.join(path_prefix, "progress.txt"), 'a') as fp:
            if not curr_order == 0:
                fp.write("\n")
            fp.write(str(curr_duration))
    print("NEXT DURATION: ", next_duration)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path-prefix', help='dataset parent folder', default='./pilot_study')
    parser.add_argument('-w', '--warmup', help='warmup', action='store_true')
    parser.add_argument('-m', '--main-collect' ,help='main collection', action='store_true')
    parser.add_argument('-cam', '--camera', help='the camera used for video capturing', type=int, default=0)
    
    args = parser.parse_args()
    speed_study(args.camera, args.path_prefix, args.warmup, args.main_collect)