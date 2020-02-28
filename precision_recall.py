import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1)) 

x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
# todo lo que es diferente de 1 queda marcado como 0
y_train[y_train!=1]=0
y_test[y_test!=1]=0

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)

# pueden ser complejos por baja precision numerica, asi que los paso a reales
valores = np.real(valores)
vectores = np.real(vectores)

# reordeno de mayor a menor
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

x_train = x_train @ vectores
x_test = x_test @ vectores

i = 10
clf = LinearDiscriminantAnalysis()
clf.fit(x_train[:,:i], y_train)
ypred_train = clf.predict(x_train[:,:i])
ypred_test = clf.predict(x_test[:,:i])
proba = clf.predict_proba(x_test[:,:i])

precision, recall, thresholds = precision_recall_curve(y_test, proba[:,1])
f1 = 2*precision*recall /(precision+recall)

n_max = f1.argmax()
plt.figure(figsize=(9,9))

plt.subplot(3,2,5)
plt.scatter(thresholds, f1[1:], marker='.')
plt.scatter(thresholds[n_max], max(f1),c='red')
plt.xlabel('probabilidad')
plt.ylabel('F1')

plt.subplot(3,2,6)
plt.plot(recall, precision)
plt.scatter(recall[n_max], precision[n_max],c='red')
plt.xlabel('recall')
plt.ylabel('precision')
#plt.show()


numero = 0
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)

# pueden ser complejos por baja precision numerica, asi que los paso a reales
valores = np.real(valores)
vectores = np.real(vectores)

# reordeno de mayor a menor
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

x_train = x_train @ vectores
x_test = x_test @ vectores

i = 10
clf = LinearDiscriminantAnalysis()
clf.fit(x_train[:,:i], y_train)
ypred_train = clf.predict(x_train[:,:i])
ypred_test = clf.predict(x_test[:,:i])
proba = clf.predict_proba(x_test[:,:i])

precision_2, recall_2, thresholds_2 = precision_recall_curve(y_test, proba[:,1])
f1_2 = 2*precision_2*recall_2 /(precision_2+recall_2)

n_max_2 = f1_2.argmax()
plt.figure(figsize=(9,9))

plt.subplot(3,2,3)
plt.scatter(thresholds_2, f1_2[1:], marker='.')
plt.scatter(thresholds_2[n_max_2], max(f1_2),c='red')
plt.xlabel('probabilidad')
plt.ylabel('F1')

plt.subplot(3,2,4)
plt.plot(recall_2, precision_2)
plt.scatter(recall_2[n_max_2], precision_2[n_max_2],c='red')
plt.xlabel('recall')
plt.ylabel('precision')
#plt.show()


cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)

# pueden ser complejos por baja precision numerica, asi que los paso a reales
valores = np.real(valores)
vectores = np.real(vectores)

# reordeno de mayor a menor
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

x_train = x_train @ vectores
x_test = x_test @ vectores

i = 10
clf = LinearDiscriminantAnalysis()
clf.fit(x_train[:,:i], y_train)
ypred_train = clf.predict(x_train[:,:i])
ypred_test = clf.predict(x_test[:,:i])
proba = clf.predict_proba(x_test[:,:i])

precision, recall, thresholds = precision_recall_curve(y_test, proba[:,1])
f1 = 2*precision*recall /(precision+recall)

n_max = f1.argmax()
plt.figure(figsize=(9,9))

plt.subplot(3,2,5)
plt.scatter(thresholds, f1[1:], marker='.')
plt.scatter(thresholds[n_max], max(f1),c='red')
plt.xlabel('probabilidad')
plt.ylabel('F1')

plt.subplot(3,2,6)
plt.plot(recall, precision)
plt.scatter(recall[n_max], precision[n_max],c='red')
plt.xlabel('recall')
plt.ylabel('precision')
plt.savefig('F1_prec_recall.png')
plt.show()
