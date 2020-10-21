import numpy as np 
import matplotlib.pyplot as plt 


def decodage(array):
    result = []
    for i in range(len(array[0])):
        sub_result = []
        for j in range(0,4) : 
            a = array[j][i].tolist()
            maxi = np.max(a)
            index = a.index(maxi)
            if j == 0 : #head 
                if index == 0 : 
                    sub_result.append('Vertex')
                elif index == 1  : 
                    sub_result.append('Eye')
                else : 
                    sub_result.append('Mouth')
            elif j == 1 : #leg 
                if index == 0 : 
                    sub_result.append('Hips')
                elif index == 1 : 
                    sub_result.append('Knee')
                else : 
                    sub_result.append('Foot')

            elif j == 2 : #right arm 
                if index == 0 : 
                    sub_result.append('down')
                else : 
                    sub_result.append('up')

            elif j == 3 : #left arm
                if index == 0 : 
                    sub_result.append('down')
                else : 
                    sub_result.append('up')
        result.append(sub_result)
    return result

def affichage(liste_array, liste_label):
    for i in range(len(liste_array)):
        image = liste_array[i][:,:,0]
        f = plt.figure(figsize=(5,8))
        axes = plt.gca()
        axes.set_axis_off()
        plt.imshow(image, cmap='gray')
        plt.title(liste_label[i])
        plt.show()
        filename = '/home/deeplearning/Deep_Learning/classification/test/classic/predictions'+'/'+str(i)+'.png'
        f.savefig(filename, bbox_inches='tight')
