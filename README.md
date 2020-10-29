# ML_Attack_XOR_PUF
The ML_Attack_XOR_PUF is a Machine Learning-based MLP model for attacking the XOR Physical Unclonable Functions using a small number of challenge-response-pair and short computation time.

The project is written using Python 3.7 using keras library with Tensor Flow backend.

Before you run the current version of attacking the 9-XPUF:
Please unzip /CRPs/challenges_9XOR_64bit_LUT_2239B_attacking_5M.mymemmap.zip then run!

Before you attack using your data:
  1- choose the correct stream value:
    The stream variable in the parameter indicates the number of components in an n-XPUF. It is used in the program to create the structure of the NN 
    (2^n/2, 2^n, 2^n/2)
    However, we suggest to choose stream = 5 if you are attacking the 2-XPUF, 3-XPUF, and 4-XPUF.
    If you are attacking the 9-XPUF, the stream should equal eight since (2^9) neurons will lead to overfitting.
  2- Make sure you obtain the correct order of the stages:
    The challenge bits should be reversed before applying the cumulative product. Please see the line to 23 in /features/__init__.py/
    If your challenges are already in reverse order, please delete line 23 in /features/__init__.py/. (You may try both orders for verification).
  3- Make sure that your challenges are randomly selected and uniformly distributed. 
    
    

The full dataset of the research can be downloaded from:
https://www.kaggle.com/khalidtmursi/attacking-xor-physical-unclonable-function-crps

Please cite the following paper:

Mursi, K.T.; Thapaliya, B.; Zhuang, Y.; Aseeri, A.O.; Alkatheiri, M.S. A Fast Deep Learning Method for Security Vulnerability Study of XOR PUFs. Electronics 2020, 9, 1715

The paper can be found at:
https://www.mdpi.com/2079-9292/9/10/1715




