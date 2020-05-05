
import numpy as np


'''
conver the challenges from hex to bin
'''
def hexToBinary(challenge):
    binChallenge=''
    for c in challenge:
        # print(h)
        cha = bin(int('0x'+c, 16)).split('b')
        binChallenge += cha[1].zfill(4)
    return binChallenge


'''
apply the cumulative product on the challenges
'''
def transform_features(C):
    # convert into 1 and -1
    C = 2. * C - 1
    # cumulative product
    C = np.cumprod(C, axis=1,  dtype=np.int8)
    return C