#HÜSEYİN HÜRKAN KARAMAN 1030516163 1.Öğretim A grubu
#SEMİH CAN BOZOK 1030516177 1.Öğretim A Grubu
#TIMAN AHMAT DJEROU 1030516145 1.Öğretim A Grubu

from __future__ import division, print_function
import csv, os, sys
import numpy as np
from SVM import SVM
import random
filepath = os.path.dirname(os.path.abspath(__file__))

def readData(filename, header=True):
    data, header = [], None
    with open(filename, 'rt',encoding = 'UTF8') as csvfile:#dosya okuması burada gerçekleşiyor. readtext
        spamreader = csv.reader(csvfile, delimiter=',')
        if header:
            header = spamreader.next()
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))

def calc_acc(y, y_hat):
    idx = np.where(y_hat == 0)
    TZ = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    return float(TZ + TP)/len(y)

def main(filename='data/data_banknote_authentication.txt', C=1.0, kernel_type='linear', epsilon=0.001):
    # Load data
    (data, _) = readData('%s/%s' % (filepath, filename), header=False)
    data = data.astype(float)
    
    random.shuffle(data)#data karıştırılıyor rastgele
    train_data = data[:int((len(data)+1)*.80)] #Remaining 80% to training set
    test_data = data[int(len(data)*.80+1):] #Splits 20% data to test set
    
    #Data parçalama
    X, y = train_data[:,0:-1], train_data[:,-1].astype(int)#Eğitim için
    X1, y1 = test_data[:,0:-1], test_data[:,-1].astype(int)#Test için
    

    # Split data
    #X, y = data[:,0:-1], data[:,-1].astype(int)

    # Initialize model
    model = SVM()

    # Fit model
    support_vectors, iterations = model.fit(X, y)#eğitim için olan datalar burada gidiyor

    # Support vector count
    sv_count = support_vectors.shape[0]

    # Make prediction
    y_hat = model.predict(X1)#test için datalar burada kullanılıyor

    # Calculate accuracy
    acc = calc_acc(y1, y_hat)

    print("Support vector count: %d" % (sv_count))
    print("bias:\t\t%.3f" % (model.b))
    print("w:\t\t" + str(model.w))
    print("accuracy:\t%.3f" % (acc))
    print("Converged after %d iterations" % (iterations))

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("")
        print("Trains a support vector machine.")
        print("Usage: %s FILENAME C kernel eps" % (sys.argv[0]))
        print("")
        print("FILENAME: Relative path of data file.")
        print("C:        Value of regularization parameter C.")
        print("kernel:   Kernel type to use in training.")
        print("          'linear' use linear kernel function.")
        print("          'quadratic' use quadratic kernel function.")
        print("eps:      Convergence value.")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['filename'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['C'] = float(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['kernel_type'] = sys.argv[3]
        if len(sys.argv) > 4:
            kwargs['epsilon'] = float(sys.argv[4])
        if len(sys.argv) > 5:
            sys.exit("Not correct arguments provided. Use %s -h for more information"
                     % (sys.argv[0]))
        main(**kwargs)
