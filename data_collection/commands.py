'''
Command sets
2/18/2021, Ruidong Zhang, rz379@cornell.edu
'''

import cv2
import math
import random
import numpy as np

# from generate_sentences import generate_sentences

cmd_digits = {
    'cmds': [
        'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine',
    ],
    'instruction_imgs': []
}


cmd_touch = {
    'cmds': [
        '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ],
    'instruction_imgs': []
}

cmd_5fingers = {
    'cmds': [
        '1', '2', '3', '4', '5',
    ],
    'instruction_imgs': [
        'ft1.png', 'ft2.png', 'ft3.png', 'ft4.png', 'ft5.png'
    ]
}

cmd_wristmotion = {
    'cmds': [
        'wrist_up', 'wrist_down', 'wrist_left',
    ],
    'instruction_imgs': [
        'wristup.png', 'wristdown.png', 'wristleft.png',
    ]
}

cmd_ftfingers = {
    'cmds': [
        'No', '1', '2', '3', '4', '5',
        '12', '23', '34', '45',
        '123', '234', '345',
        '1234', '2345', '12345',
    ],
    'instruction_imgs': [
        'ft0.png', 'ft1.png', 'ft2.png', 'ft3.png', 'ft4.png', 'ft5.png',
        'ft12.png', 'ft23.png', 'ft34.png', 'ft45.png',
        'ft123.png', 'ft234.png', 'ft345.png',
        'ft1234.png', 'ft2345.png', 'ft12345.png',
    ]
}

cmd_asldigits = {
    'cmds': [
        'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine',
    ],
    'instruction_imgs': [
        'asl0.png', 'asl1.png', 'asl2.png', 'asl3.png', 'asl4.png', 'asl5.png', 'asl6.png', 'asl7.png', 'asl8.png', 'asl9.png',
    ]
}

cmd_43 = {
    'cmds': [
        'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine',
        'Weather', 'News', 'Alarm', 'Time', 'Traffic', 'Camera',
        'Previous', 'Next', 'Pause', 'Resume', 'Stop', 'Volume', 'Up', 'Down',
        'Message', 'Send', 'Hang up',
        'Answer', 'Call', 'Check', 'Copy', 'Cut', 'Help', 'Home', 'Mute',
        'Paste', 'Play', 'Redial', 'Screenshot', 'Search', 'Skip', 'Skype', 'Undo',
    ],
    'instruction_imgs': []
}

cmd_music = {
    'cmds': [
        'Previous', 'Next', 'Pause', 'Resume', 'Stop', 'Volume up', 'Volume down', 'Play',
        # 'What\'s the weather', 'Latest news', 'Set an alarm', 'What time is it', 'How\'s the traffic', 'Open camera', 'Hang up',
    ],
    'instruction_imgs': []
}

cmd_int = {
    'cmds': [
        'What\'s the weather', 'Latest news', 'Set an alarm', 'What time is it', 'How\'s the traffic', 'Open camera', 'Hang up',
    ],
    'instruction_imgs': []
}

cmd_speechin = {
    'cmds': [
        'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine',
        'Answer', 'Call', 'Camera', 'Check', 'Copy', 'Cut', 'Hang up', 'Help', 'Home', 'Mute',
        'Paste', 'Pause', 'Play', 'Previous', 'Redial', 'Screenshot', 'Search', 'Skip', 'Skype', 'Undo',
        'Volume', 'Share', 'Next', 'Open', 'Close', 'Keyboard',
        'OK Google', 'Hey Siri', 'Alexa',
        'Question mark', 'Exclamation point', 'Comma', 'Dot', 'Semicolon', 'Colon', 'Quotation mark',
        'Parentheses', 'Dash', 'Slash', 'Underscore',
        'Left', 'Right', 'Up', 'Down',
    ],
    'instruction_imgs': []
}

cmd_punctuations = {
    'cmds': [
        # 'Comma', 'Semicolon', 'Colon', 'Parentheses', 'Dash', 'Slash', 'At', 'Dollar', 'Asterisk',
        # 'Underscore', 'Period', 'Question mark', 'Exclamation mark', 'Quotation mark',
        'Colon', 'Dash', 'At', 'Dollar', 'Asterisk', 'Equal',
    ],
    'instruction_imgs': []
}

# cmd_punctuations = {
#     'cmds': [
#         'Question mark', 'Exclamation mark', 'Comma', 'Semicolon', 'Colon', 'Quotation mark',
#         'Parentheses', 'Dash', 'Slash', 'Underscore',
#     ],
#     'instruction_imgs': []
# }

cmd_31 = {
    'cmds': [
        'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine',
        'Play', 'Stop', 'Resume', 'Pause', 'Previous', 'Next', 'Volume',
        'Left', 'Right', 'Up', 'Down', 'OK', 'Cancel', 'Menu', 'Dial', 'Hang up', 'Open', 'Close',
        'Hey Google', 'Hey Siri', 'Alexa',
    ],
    'instruction_imgs': []
}

cmd_mp = {
    'cmds': [
        # 'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine',
        'Play', 'Stop', 'Resume', 'Pause', 'Previous', 'Next', 'Volume up', 'Volume down',# 'Down',
        # 'Left', 'Right', 'Up', 'Down', 'OK', 'Cancel', 'Menu', 'Dial', 'Hang up', 'Open', 'Close',
        # 'Hey Google', 'Hey Siri', 'Alexa',
    ],
    'instruction_imgs': []
}

cmd_cad = {
    'cmds': [
        'Rectangle', 'Circle', 'Tab', 'Unlock',
    ],
    'instruction_imgs': []
}

cmd_tongue = {
    'cmds': [
        'Front', 'Back', 'Left', 'Right', 'Tap',
    ],
    'instruction_imgs': []
}

cmd_objs = {
    'cmds': [
        'obj1', 'obj2', 'obj3', 'obj4', 'obj5', 'obj6', 'obj7', 'obj8', 'obj9', 
    ],
    'instruction_imgs': [
        'obj_1.png', 'obj_2.png', 'obj_3.png', 'obj_4.png', 'obj_5.png', 'obj_6.png', 'obj_7.png', 'obj_8.png', 'obj_9.png'
    ]
}

cmd_chi = {
    'cmds': [
        'Zero', 'Two', 'Four',
    ],
    'instruction_imgs': [
        'asl0.png', 'asl2.png', 'asl4.png',
    ]
}

cmd_chi_gesture = {
    'cmds': [
        '1', '5', 'One', 'Three', 'Seven', 'Eight', 'Nine'
    ],
    'instruction_imgs': [
        'ft1.png', 'ft5.png', 'asl1.png', 'asl3.png', 'asl7.png', 'asl8.png', 'asl9.png',
    ]
}

cmd_flute = {
    'cmds': [
        '2', 'Zero', 'A', 'G', 'null'   # '3', '4', '5',
    ],
    'instruction_imgs': [
        'flute_1.png', 'flute_5.png', 'flute_6.png', 'flute_7.png', 'asl5.png'  # 'flute_2.png', 'flute_3.png', 'flute_4.png', 
    ]
}

def generate_discrete(cmds, folds, reps_per_fold):

    original_cmd_list = []
    img_list = {}
    for i, cmd in enumerate(cmds['cmds']):
        # manually assigned duration
        # if len(cmd.split(' ')) > 1:
        #     duration = 1.6
        # else:
        #     duration = 1.4 + random.random() * 0.2
        
        if len(cmds['instruction_imgs']) == 0:
            original_cmd_list += [(i, cmd, None)]
        else:
            original_cmd_list += [(i, cmd, cmds['instruction_imgs'][i])]
            img_list[cmds['instruction_imgs'][i]] = cv2.resize(cv2.imread('img/%s' % cmds['instruction_imgs'][i]), (331, 400))

    cmd_list = []
    cmds_repped = original_cmd_list * reps_per_fold
    for _ in range(folds):
        random.shuffle(cmds_repped)
        cmd_list += cmds_repped

    return cmd_list, img_list


def generate_echowrist(folds, reps_per_fold):
    fingers_cmds, fingers_img = generate_discrete(cmd_5fingers, folds, reps_per_fold)
    asldigits_cmds, asldigits_img = generate_discrete(cmd_asldigits, folds, reps_per_fold)
    wristmotion_cmds, wristmotion_img = generate_discrete(cmd_wristmotion, folds, reps_per_fold)

    for i in range(len(asldigits_cmds)):
        asldigits_cmds[i] = (asldigits_cmds[i][0] + 5, asldigits_cmds[i][1], asldigits_cmds[i][2])
    
    for i in range(len(wristmotion_cmds)):
        wristmotion_cmds[i] = (wristmotion_cmds[i][0] + 5 + 10, wristmotion_cmds[i][1], wristmotion_cmds[i][2])

    imgs = {}
    for k, v in fingers_img.items():
        imgs[k] = v
    for k, v in asldigits_img.items():
        imgs[k] = v
    for k, v in wristmotion_img.items():
        imgs[k] = v

    return fingers_cmds + [[10, 'Five', 'asl5.png']] + asldigits_cmds + [[10, 'Five', 'asl5.png']] + wristmotion_cmds, imgs

def generate_chi(folds, reps_per_fold):
    chi_cmds, chi_img = generate_discrete(cmd_chi, folds, reps_per_fold)
    # wristmotion_cmds, wristmotion_img = generate_discrete(cmd_wristmotion, folds, reps_per_fold)
    
    # for i in range(len(wristmotion_cmds)):
    #     wristmotion_cmds[i] = (wristmotion_cmds[i][0] + 3, wristmotion_cmds[i][1], wristmotion_cmds[i][2])

    imgs = {}
    for k, v in chi_img.items():
        imgs[k] = v
    # for k, v in wristmotion_img.items():
    #     imgs[k] = v

    # return chi_cmds + [[6, 'Four', 'asl4.png']] + wristmotion_cmds, imgs
    return chi_cmds, imgs

def generate_flute(folds, reps_per_fold):
    flute_cmds, flute_img = generate_discrete(cmd_flute, folds, reps_per_fold)

    imgs = {}
    for k, v in flute_img.items():
        imgs[k] = v

    return flute_cmds, imgs

def generate_wristmotions(folds, reps_per_fold):
    wristmotion_cmds, wristmotion_img = generate_discrete(cmd_wristmotion, folds, reps_per_fold)

    imgs = {}
    for k, v in wristmotion_img.items():
        imgs[k] = v

    return wristmotion_cmds, imgs

def generate_chi_gesture(folds, reps_per_fold):
    chi_gesture_cmds, chi_gesture_img = generate_discrete(cmd_chi_gesture, folds, reps_per_fold)
    # wristmotion_cmds, wristmotion_img = generate_discrete(cmd_wristmotion, folds, reps_per_fold)
    
    # for i in range(len(wristmotion_cmds)):
    #     wristmotion_cmds[i] = (wristmotion_cmds[i][0] + 3, wristmotion_cmds[i][1], wristmotion_cmds[i][2])

    imgs = {}
    for k, v in chi_gesture_img.items():
        imgs[k] = v
    # for k, v in wristmotion_img.items():
    #     imgs[k] = v

    # return chi_cmds + [[6, 'Four', 'asl4.png']] + wristmotion_cmds, imgs
    return chi_gesture_cmds, imgs

def generate_connected_digits(folds, reps_per_fold, len_range=(3, 6)):
    # base_cmds = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    base_cmds = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

    cmd_list = []
    img_list = {}

    for _ in range(folds):
        len_seq = list(range(len_range[0], len_range[1] + 1))
        len_seq *= math.ceil(reps_per_fold / len(len_seq))
        random.shuffle(len_seq)
        len_seq = len_seq[:reps_per_fold]

        fold_digits_seq = list(range(len(base_cmds))) * math.ceil(np.sum(len_seq) / len(base_cmds))
        random.shuffle(fold_digits_seq)

        digit_pos = 0
        for l in len_seq:
            this_cmd_index = fold_digits_seq[digit_pos: digit_pos + l]
            this_cmd_label = ' '.join([str(x) for x in this_cmd_index])
            this_cmd_text = ' '.join([base_cmds[x] for x in this_cmd_index])
            cmd_list += [(this_cmd_label, this_cmd_text, None)]
            digit_pos += l
    return cmd_list, img_list

def generate_connected_isolated_digits(n_connected_digits=60, n_isolated_digits=81, t_cd=4, t_id=2):
    ori_cds, _ = generate_connected_digits(1, n_connected_digits)
    # ids = []
    ori_ids, _ = generate_discrete(cmd_punctuations, 1, 10)

    cds = []
    ids = []
    for l, t, _ in ori_cds:
        if random.random() > 1.2:
            l_s = l.split()
            t_s = t.split()
            point_pos = random.randint(0, len(l_s) - 2)
            l_s.insert(point_pos + 1, '10')
            t_s.insert(point_pos + 1, 'Point')
            cds += [(' '.join(l_s), ' '.join(t_s), None)]
        else:
            cds += [(l, t, None)]

    for l, t, _ in ori_ids:
        ids += [(str(int(l) + 15), t, None)]

    # c_i_ds = [(x[0], x[1], x[2], t_cd) for x in cds] + [(x[0], x[1], x[2], t_id) for x in ids]
    c_i_ds = [(x[0], x[1], x[2], t_cd) for x in cds]
    random.shuffle(c_i_ds)

    return c_i_ds, {}

# def generate_grid(folds, reps_per_fold):
#     base_words = [
#         ['Bin', 'Lay', 'Place', 'Set'],
#         ['blue', 'green', 'red', 'white'],
#         ['at', 'by', 'in', 'with'],
#         ['A', 'B', 'C', 'D'],
#         ['1', '2', '3', '4'],
#         ['again', 'now', 'please', 'soon']
#     ]

#     n_groups = len(base_words)

#     assert(reps_per_fold % 4 == 0)  # saves the trouble :)
#     reps_each_word = reps_per_fold // 4
#     base_words_labels = list(range(sum([len(x) for x in base_words])))

#     generated_rand_groups = [[] for _ in range(n_groups)]

#     for i in range(n_groups):

def generate_objs(folds, reps_per_fold):
    obj_cmds, obj_img = generate_discrete(cmd_objs, folds, reps_per_fold)

    imgs = {}
    for k, v in obj_img.items():
        imgs[k] = v

    return obj_cmds, imgs
        

def load_cmds(cmd_set, folds, reps_per_fold):

    original_cmd_list = []
    img_list = {}
    cs = cmd_set.lower().replace('_', '')
    if cs == 'digits':
        cmds = cmd_digits
    elif cs == 'speechin':
        cmds = cmd_speechin
    elif cs == '43':
        cmds = cmd_43
    elif cs == '31':
        cmds = cmd_31
    # elif cs == '15':
    #     cmds = cmd_15
    elif cs == 'music':
        cmds = cmd_music
    elif cs == 'punctuations':
        cmds = cmd_punctuations
    elif cs == 'cad':
        cmds = cmd_cad
    elif cs == 'tongue':
        cmds = cmd_tongue
    elif cs == 'int':
        cmds = cmd_int
    elif cs == 'touch':
        cmds = cmd_touch
    elif cs == '5fingers':
        cmds = cmd_5fingers
    elif cs == 'ftfingers':
        cmds = cmd_ftfingers
    elif cs == 'asldigits':
        cmds = cmd_asldigits
    elif cs == 'objs':
        cmds = cmd_objs
    elif cs == 'chi':
        cmds = cmd_chi
    elif cs == 'flute':
        cmds = cmd_flute
    elif cs not in ['connecteddigits', 'grid', 'sentences']:
        raise ValueError('Command set with name %s not found' % cmd_set)
    
    if cs == 'connecteddigits':
        return generate_connected_digits(folds, reps_per_fold, len_range=(3, 6))
    elif cs == 'sentences':
        return generate_sentences(folds, reps_per_fold)
    else:
        return generate_discrete(cmds, folds, reps_per_fold)
