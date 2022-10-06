import csv
import numpy as np

class gen_data():
    def __init__(self, set_num):
        self.set_num  = set_num
        set1_desc = """    ****************************************************************
    This dataset was collected manually for the final semester project. 
    performed at: The School of Biotechnology and Bioinformetics, DY Patil University.
    ****************************************************************"""
        if self.set_num == 'set1':
            print(set1_desc, "\n")
        else:
            print("Error! Dataset not found!")

    def get_data(self, set_name):
        if self.set_num == 'set1':
            features = np.loadtxt(r'features_fin.csv',dtype=str,delimiter=',',usecols=(range(1,9)),skiprows=1)
            features = features.astype(float)
            classes = np.loadtxt(r'features_fin.csv',dtype=str,delimiter=',',usecols=(9,),skiprows=1)
            classes = classes.astype(float)
        else:
            print("Error! Dataset not found!")
            return
        
        train_size = np.floor(0.6*len(classes))
        test_size = np.floor(0.2*len(classes))
        val_size = np.floor(0.2*len(classes))
        
        if train_size + test_size + val_size > len(classes):
            print("Error! Sum of sets cannot be larger than database \n")
        else:
            print("Successful split! processing data... \n")
        
        train_set = features[:int(train_size)]
        train_classes = classes[:int(train_size)]
        test_set = features[int(train_size):int(train_size+test_size)]
        test_classes = classes[int(train_size):int(train_size+test_size)]
        val_set = features[int(train_size+test_size):]
        val_classes = classes[int(train_size+test_size):]
        
        if set_name == 'train':
            return train_set, train_classes
        elif set_name == 'test':
            return test_set, test_classes
        elif set_name == 'val':
            return val_set, val_classes
        else:
            print("Error! Invalid set requested, please check parameters")

if __name__ == '__main__':
    datagen = gen_data()
    train_feat, train_class = datagen.get_data(set_name='train')
    print(train_feat)