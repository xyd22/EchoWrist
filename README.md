# README

# Data Collection

1. Change the working directory to data_collection

```bash
cd data-collection
```

2. Collect data

```bash
python data_collection.py -cd 6 -c 5fingers -f 2 -r 2 -t 2 -cam 0 --audio True --noserial -p ../pilot_study -o test_old_5
```

- -p: dataset parent folder
- -o: output directory name
- -cd: countdown time (s) before start
- -c: command set name, comma separated if multiple
- -f: number of folds
- -r: number of repetitions per feature for a fold
- -t: duration of each command/gesture
- -cam:  the camera used for video capturing
- --audio: Toggle audible commands (audible only if flag is set to True AND if we are using the 'objs' command set)
- --noserial: add when using the wireless prototype

# Data Processing

1. Change the working directory to data_preparation

```bash
cd data_preparation
```

2. Acoustic signal and video data syncing
    1. Copy the .raw audio file(s) from the SD card to the directory specified through the -o flag in the data collection script
    2. Find out the frame number in the recorded video file where the participant clapped
    3. Add the frame number in the “syncing_poses” list under the “ground_truth” field of config.json 

```bash
python audio_auto_sync.py --path ../pilot_study/test_old_5
```

3. Data preparation for training

```bash
python data_preparation.py -md 500000000 -nd -500000000 -f --path ../pilot_study/test_old_5
```

4. Echo profile visualization

```bash
python visualize.py --height 605 --echo_length 30 --path ../pilot_study/test_old_5
```
Includes multiple flags for more specific visualizations. Notable ones are:
- --echo_length: display length of echo_profile in video
- --height: height of the output video
- -coi: channels of interest to display in video
- -poi: pixels of interest to display in a given channel (format of arg is LOWBOUND,HIGHBOUND)
- -md: maximum value to clip values into when coloring differential plots
- -nd: minimum value to clip values into when coloring differential plots

5. Pushing  collected data to the server

# Training

1. Change the working directory to dl_model

```bash
cd dl_model
```

2. Training

```bash
python train.py -o train_output -f original-ts 0101 -p /data3/cj/Obj_Data/test_old_5 
```
- -f: train with original, diff, or both echo profiles (default: both)
