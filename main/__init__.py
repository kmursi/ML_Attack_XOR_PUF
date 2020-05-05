'change the project default path'
import sys
sys.path.append('../')

import argparse
import numpy as np
import models as md
import features as fe

"""
*************************************************************************************
*                        Example of a command to run this project                   *
*************************************************************************************
"""
'python3 __init__.py --challenges 3000 --epochs 100 --streams 5 --filename ../CRPs/5XOR_64bit.txt --plot 1'

"""
*************************************************************************************
*                                 Arguments handling                                *
*************************************************************************************
"""
def get_args():
    parser = argparse.ArgumentParser(
                        description="Experiments on XPUF with strames ranged from 4 to 9",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stages', metavar="S", type=int,
                        default=64, help='Test size (0-1)')
    parser.add_argument('--challenges', metavar="C", type=int,
                        default=40000, help='Number of challenges')
    parser.add_argument('--streams', metavar="SS", type=int, default=4, help='Number of streams in XPUF')
    parser.add_argument('--epochs', metavar="R", type=int, default=50, help='Number of epochs')
    parser.add_argument('--plot', metavar="PC", type=int, default=0, help='Plotting option')
    parser.add_argument('--plotdist', metavar="PT", nargs='?', default='loss_and_accuracy.png')
    parser.add_argument('--filename', metavar="FN", type=str,default='../CRPs/4XOR_64bit.txt', help='CRPs file path')
    parser.add_argument('--patience', metavar="P", type=int, default=5, help='Early stopping patience')

    return parser.parse_args()

"""
*************************************************************************************
*                                 Reading CRPs dataset                              *
*************************************************************************************
"""
'This function read and split the CRPs of the dataset and store them in a predefined arrays'
def reading_file(args, challenges_array, response_array):
    print('********** Start rading file **********')
    with open(args.filename, 'r', buffering=1) as infile:
        i = 0
        for line in infile:
            sp = line.split(';')
            temp = np.asanyarray([x for x in fe.hexToBinary(sp[0].strip())])
            challenges_array[i] = temp
            res1 = sp[1].strip()
            if res1 == '0':
                response_array[i] = np.int8(-1)
            else:
                response_array[i] = np.int8(1)

            i = i + 1
            if (i == args.challenges): break

    print('file ', args.filename, ' read successfully !')
    print('\n')

    return challenges_array, response_array

"""
*************************************************************************************
*                    Model transformation and training                              *
*************************************************************************************
"""
def model_training(args, model):
    challenges_array = np.asanyarray([[0 for x in range(args.stages)] for y in range(args.challenges)])
    response_array = np.asanyarray([0 for x in range(args.challenges)])

    challenges_array, response_array = reading_file(args, challenges_array, response_array)

    print('********** Start challenges transformation **********')
    challenges_array = fe.transform_features(challenges_array)
    print('Sample of the transformed challenges:')
    print(challenges_array[0:5])
    print('\n')

    print('********** Start training model **********')
    model.fit(challenges_array, response_array)


"""
*************************************************************************************
*                                   Main function                                   *
*************************************************************************************
"""

if __name__ == "__main__":
    '''
    PART_1:
    Start initialization
    '''
    args = get_args()
    print('\n')
    print('********** Start initialization **********')
    print(args)
    print('\n')
    model = md.XOR_PUF(stages=args.stages, streams=args.streams, epochs=args.epochs, plot=args.plot, fig_name=args.plotdist)

    '''
    PART_2:
    Model Training
    '''
    model_training(args, model)


