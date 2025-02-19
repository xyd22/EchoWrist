import argparse
from train import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand pose from mutliple views')
    parser.add_argument('-p', '--print-freq', default=30, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('-v', '--valid-freq', default=3, type=int,
                        help='validation frequency (default: 5)')
    parser.add_argument('-o', '--output', default='temp', type=str,
                        help='the name of output file')
    parser.add_argument('-i', '--input', default='', type=str,
                        help='the name of input file')
    parser.add_argument('-n', '--name', default='', type=str,
                        help='the name of dataset (overwrites config file)')
    parser.add_argument('-ts', '--test-sessions', default='', type=str,
                        help='sessions to be used for testing, comma separated, e.g. 5,6,7')
    parser.add_argument('-tp', '--test-participants', default='', type=str,
                        help='sessions to be used for testing, comma separated, e.g. P5,P6,P7')
    parser.add_argument('--train-sessions', default='', type=str,
                        help='sessions to be used for training, default using all but testing sessions for training, comma separated, e.g. 5,6,7')
    parser.add_argument('-g', '--visible-gpu', default='', type=str,
                        help='visible gpus, comma separated, e.g. 0,1,2 (overwrites config file)')
    # parser.add_argument('-m', '--mode', default='', type=str,
    #                     help='the mode of training')
    parser.add_argument('--epochs', default=-1, type=int,
                        help='total epochs to run')
    parser.add_argument('--epochs-to-run', default=-1, type=int,
                        help='how many epochs left to run, overwrites --epochs')
    parser.add_argument('--lr', default=0, type=float,
                        help='learning rate')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-t', '--if_train', default=0, type=int,
                        help='If train this model (default: none)')
    parser.add_argument('-f', '--training-file', default='both', type=str,
                        help='Train with original, diff, or both echo profiles (default: both)')
    args = parser.parse_args()

    all_participants = ['P%d' % i for i in range(1, 12 + 1)]
    train_p = []
    test_p = args.test_participants.split(',')
    for p in all_participants:
        if p not in test_p:
            train_p += [p]

    train_sessions = []
    test_sessions = []
    for r in range(1, 4 + 1):
        for s in range(1, 5 + 1):
            for p in train_p:
                train_sessions += ['%s_%02d%d' % (p, r, s)]
            for p in test_p:
                test_sessions += ['%s_%02d%d' % (p, r, s)]

    train_sessions.sort()
    test_sessions.sort()
    args.train_sessions = ','.join(train_sessions)
    args.test_sessions = ','.join(test_sessions)
    main(args)