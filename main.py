import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy as Scipy
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import eegtools.spatfilt as csp
import random



def mFiltCSP(X, nF):
    C1 = X[0]
    C2 = X[1]
    for i in range(2, len(X)):
        if X[i][0,-1] == 1:
            C1 = np.concatenate((C1,X[i]), axis=0)
        if X[i][0,-1] == 2:
            C2 = np.concatenate((C2,X[i]), axis=0)

    entrada1 = np.cov(np.transpose(C1[:,:13]))
    entrada2 = np.cov(np.transpose(C2[:,:13]))
    W = csp.csp(entrada1, entrada2, nF)
    X = 0
    entrada1 = 0
    entrada2 = 0
    C1 = 0
    C2 = 0

    return W

def mProcCSP(X, W):
    colClass = np.zeros((len(X), 2))
    colClass[0,:] = X[0][0,-2:]

    dataN = np.matrix(X[0][:,:13]) * np.matrix(np.transpose(W))
    colDataFil = np.var(dataN, axis=0)
    for i in range(1, len(X)):
        colClass[i,:] = X[i][0,-2:]
        dataN = np.matrix(X[i][:,:13]) * np.matrix(np.transpose(W))
        colDataFil=  np.concatenate((colDataFil, np.var(dataN, axis=0)),axis=0)
        dataN = 0

    colDataFil = np.concatenate((colDataFil, colClass), axis=1)
    print np.shape(colDataFil)
    return colDataFil


inicioSeg = 200
finSeg = 1200
n_samples = 250.0
nFiltros = 13

data = np.loadtxt("data/B0101TDef2.dat", delimiter=",")
dataF = []
dataTemporal = data[np.where(data[:,6] == 1),:]
dataTemporal = dataTemporal[0]
dataF.append(dataTemporal[inicioSeg:finSeg, :])

cantidadEnsayos = int(max(data[-2]))

for i in range(2,  cantidadEnsayos+1):
    dataTemporal = data[np.where(data[:,6] == i),:]
    dataTemporal = dataTemporal[0]
    dataTemporal = dataTemporal[inicioSeg:finSeg, :]
    dataF.append(dataTemporal)


dataTrain = dataF
#Filtrado

# dataF = 0

colFiltradoPB = []
for i in range(1, cantidadEnsayos +1):


    datosConca = np.array(dataTrain[i-1][np.where(dataTrain[i-1][:,-2] == i), -2:])
    datapFiltrar = np.array(dataTrain[i-1][np.where(dataTrain[i-1][:,-2] == i),:3])
    (b, a) = signal.butter(3, np.array([4, 8]) / (n_samples / 2), 'band')
    filtrado0 = signal.lfilter(b, a, datapFiltrar , 0)
    (b, a) = signal.butter(3, np.array([8, 15]) / (n_samples / 2), 'band')
    filtrado1 = signal.lfilter(b, a, datapFiltrar , 0)
    (b, a) = signal.butter(3, np.array([15, 30]) / (n_samples / 2), 'band')
    filtrado2 = signal.lfilter(b, a, datapFiltrar , 0)
    (b, a) = signal.butter(3, np.array([30, 40]) / (n_samples / 2), 'band')
    filtrado3 = signal.lfilter(b, a, datapFiltrar , 0)


    colFiltradoPB.append(np.concatenate((filtrado0[0], filtrado1[0], filtrado2[0], filtrado3[0], datosConca[0]), axis=1))

    datosConca = 0
    datapFiltrar = 0



filtroCSP = mFiltCSP(colFiltradoPB, nFiltros)
colFiltradoCSP = mProcCSP(colFiltradoPB, filtroCSP)


#IMPREsION DE GRAFICAS LINEALES

plt.subplot(3,1,1)
p=plt.plot(colFiltradoCSP[:,[0,1,2,3,4,-1]])

plt.subplot(3,1,2)
p=plt.plot(colFiltradoCSP[:,[5,6,7,8,9,10,11,12,-1]])
plt.show()


colFiltradoCSP = np.random.permutation(colFiltradoCSP)



C=1
while C>0.0001:
    colFiltradoCSP = np.random.permutation(colFiltradoCSP)

    #CLASIFICACION

    xT = colFiltradoCSP[:,:5]
    yT = colFiltradoCSP[:,-1]
    yT = np.ravel(yT)
    svmC = svm.SVC(kernel="poly", C=C, degree=3)
    rnC = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=1)
    lda = LinearDiscriminantAnalysis()
    scoreSvm = cross_val_score(svmC, xT, yT, cv=2, scoring="f1_macro")
    scoreRnc = cross_val_score(rnC, xT, yT, cv=2, scoring="f1_macro")
    scoreLda = cross_val_score(lda, xT, yT, cv=2, scoring="f1_macro")

    print "SVM: ",str(scoreSvm)
    print "Redes Neuronales: ",str(scoreRnc)
    print "Discriminante Lineal: ",str(scoreLda)
    print "================================="
    C = float(C) / 2


#IMPRESION DE GRAFICAS DE DISTRIBUCION
# labels0=graf1[:,-1]
# i0=np.where(labels0==1)
# i1=np.where(labels0==2)
#
# labels1=graf2[:,-1]
# wi0=np.where(labels1==1)
# wi1=np.where(labels1==2)
#
# labels2=graf3[:,-1]
# w0=np.where(labels2==1)
# w1=np.where(labels2==2)
#
# plt.subplot(2,2,1)
# p=plt.plot(graf1[i0,0],graf1[i0,1],'bo')
# p=plt.plot(graf1[i1,0],graf1[i1,1],'ro')
#
#
# plt.subplot(2,2,2)
# p=plt.plot(graf2[wi0,0],graf2[wi0,1],'bo')
# p=plt.plot(graf2[wi1,0],graf2[wi1,1],'ro')
#
#
# plt.subplot(2,2,3)
# p=plt.plot(graf3[w0,0],graf3[w0,1],'bo')
# p=plt.plot(graf3[w1,0],graf3[w1,1],'ro')
#
# plt.show()
