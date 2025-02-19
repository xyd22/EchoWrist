'''
Parse record file for corrupted gestures, return a list of corrupted time periods
Ruidong Zhang, rz379@cornell.edu, 11/15/2022
'''


'''
record_20220818_131409_438453.mp4:   
record_20220818_131814_277206.mp4: 
record_20220818_132138_619844.mp4: 
record_20220818_132526_419525.mp4: 668,702
record_20220818_132912_937088.mp4: 2944,2989
record_20220818_133219_279509.mp4: 1371,1403
record_20220818_133531_382812.mp4: 3412,3444
record_20220818_133838_902039.mp4: 3338,3382
record_20220818_134208_930067.mp4: 2143,2177
record_20220818_134512_264839.mp4: 2731,2782
record_20220818_134822_874086.mp4: 1510,1546
record_20220818_135140_507158.mp4: 
record_20220818_135436_005545.mp4: 
record_20220818_135839_483730.mp4: 
record_20220818_140158_776014.mp4: 1257,1301
record_20220818_140513_169999.mp4: 
record_20220818_140823_549974.mp4: 2496,2526; 4032,4073
record_20220818_141134_551529.mp4: 
record_20220818_141545_739654.mp4: 
record_20220818_141949_630928.mp4: 
record_20220818_142553_900485.mp4: 
record_20220818_142904_290399.mp4: 
record_20220818_143217_911016.mp4: 
record_20220818_143525_148815.mp4: 
record_20220818_144223_377681.mp4: 407,476; 2651,2689; 5189,5214
record_20220818_144608_156569.mp4: 1132,1180; 
record_20220818_144920_445138.mp4: 747,789; 
record_20220818_145435_896263.mp4: 1547,1597
record_20220818_145749_105547.mp4: 
record_20220818_150116_202362.mp4: 437,459; 888,912; 1485,1512; 2513,2551; 4110,4132
record_20220818_150437_529273.mp4: 866,880; 1818; 1850
record_20220818_150810_832670.mp4: 474,507; 3887,3900
record_20220818_151137_008457.mp4: 
record_20220818_151622_533439.mp4: 528,557; 1060,1078; 1801,1841; 2900,2917; 4053,4080; 5146,5173
record_20220818_151956_885103.mp4: 1261,1300; 2002,2079; 2988,3017; 3782,3814
record_20220818_152320_946356.mp4: 
record_20220818_152638_567428.mp4: 720,763; 1447,1494
record_20220818_153003_428984.mp4: 
[end]
'''

import os

from utils import load_frame_time, load_config

def parse_corrupted_record(dataset_parent, config_file):

    config = load_config(dataset_parent, config_file)

    # not specified, return empty list (nothing to remove)
    if 'corrupted_record' not in config['ground_truth']:
        return []
    
    record_file = config['ground_truth']['corrupted_record']

    corrupted_ts_ranges = []

    with open(os.path.join(dataset_parent, record_file), 'rt') as f:
        for l in f.readlines():
            if l.rstrip('\n') == '[end]':
                continue
            filename, corrupted_frame_ranges = l.rstrip('\n').replace(' ', '').split(':')
            if len(corrupted_frame_ranges) == 0:
                continue
            ts_file = os.path.join(dataset_parent, filename.replace('.mp4', '') + '_frame_time.txt')
            if not os.path.exists(ts_file):
                print('Warning: %s not found' % ts_file)
                continue
            video_frame_ts = load_frame_time(ts_file)
            for corrupted_frame_range in corrupted_frame_ranges.split(';'):
                if len(corrupted_frame_range) == 0:
                    continue
                frame_s, frame_e = corrupted_frame_range.split(',')
                ts_s = video_frame_ts[int(frame_s) - 1]
                ts_e = video_frame_ts[int(frame_e) - 1]
                corrupted_ts_ranges += [(ts_s, ts_e)]
    # print(corrupted_ts_ranges)
    return corrupted_ts_ranges