from scipy.io import loadmat
from numpy import array, dtype, ndarray, reshape, transpose, uint8, vectorize
from sys import argv
from matplotlib import pyplot as plt

class LetterDataset:
    #training_images
    #training_labels
    #
    
    def __init__(self, fname:str) -> None:
        m=loadmat(fname)
        md = m['dataset']
        mdtr = md['train'][0][0]
        mdte = md['test'][0][0]
        mdmp = md['mapping'][0][0]

        self.train_images = mdtr['images'][0][0]
        self.train_labels = mdtr['labels'][0][0]
        self.train_writers = mdtr['writers'][0][0]
        
        self.test_images = mdte['images'][0][0]
        self.test_labels = mdte['labels'][0][0]
        self.test_writers = mdte['writers'][0][0]

        self.mapping = { i[0]:set(i[1:]) for i in mdmp}
        
        pass
    pass

def main(argv: list[str]):
    l=LetterDataset("matlab/emnist-letters.mat")
    # print(reshape(l.test_images[0],(-1,28)))
    k=10004
    print(l.test_labels[k])
    plt.imshow(transpose(reshape(l.test_images[k],(-1,28))), interpolation='nearest')
    plt.show()
    pass

if __name__=='__main__':
    main(argv)