import numpy as np
 
softmax_output = np.array([0.7,0.2,0.3],
                        [0.1,0.4,0.5]
                        [0.09,0.8,0.2])

class_target = [1,0,0]


print(softmax_output[range(len(softmax_output)),class_target])

 
 