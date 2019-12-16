
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import \
    train_test_split  # some documents still include the cross-validation option but it no more exists in version 18.0
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
data = pd.read_csv("Dataset.csv")
print("datam=",data)
print("###########################################################################")
from scipy.stats import zscore, shapiro

def variat(List):
    a=max(List)
    b=min(List)
    c=len(List)
    List_new=[]
    rng=a-b

    for e in range(c):

        z=((List[e]-b)/rng)
        List_new.append(z)


    return List_new





print(data["R"].describe())
print(data["L"].describe())
print(data["BT"].describe())
print(data["V"].describe())
print(data["AR"].describe())
print(data["HD"].describe())
plt.title('R table')
plt.hist(data["R"])
plt.show()
plt.title('L table')
plt.hist(data["L"])
plt.show()
plt.title('BT table')
plt.hist(data["BT"])
plt.show()
plt.title('V table')
plt.hist(data["V"])
plt.show()
plt.title('AR Table')
plt.hist(data["AR"])
plt.show()
plt.title('HD Table')
plt.hist(data["HD"])
plt.show()

encoder = preprocessing.OneHotEncoder()
encoder.fit(data[["BT"]])
encoded_vectorbt = encoder.transform(data[["BT"]]).toarray()
print("encoded vectorbt",encoded_vectorbt)
print("encoded vectorun ilk satiribt",encoded_vectorbt)
data["BT"]=encoded_vectorbt

encoder2 = preprocessing.OneHotEncoder()
encoder2.fit(data[["V"]])
encoded_vectorv = encoder2.transform(data[["V"]]).toarray()
print("encoded vectorv",encoded_vectorv)
print("encoded vectorun ilk satiriv",encoded_vectorv)
data["V"]=encoded_vectorv

encoder3 = preprocessing.OneHotEncoder()
encoder3.fit(data[["AR"]])
encoded_vectorar = encoder3.transform(data[["AR"]]).toarray()
print("encoded vectorar",encoded_vectorar)
print("encoded vectorun ilk satirlari",encoded_vectorar)
data["AR"]=encoded_vectorar

stat, p = shapiro(data["R"])#her sutun icin numeric degerler ilk 2 tanesinde r ve l sadece numeric
if p>0.05:
    data["R"] = zscore(data["R"])
    print("normal")
else:
    print("not normal")
    data["R"]=variat(data["R"])

stat, p = shapiro(data["L"])  # her sutun icin numeric degerler ilk 2 tanesinde r ve l sadece numeric
if p > 0.05:
    data["L"]=zscore(data["L"])
    print("normal")
else:
    print("not normal")
    data["L"]= variat(data["L"])

    print("After=", data["L"])
    #son sutun haric x (son sutun)y
X_train, X_test, y_train, y_test = train_test_split(data[["R", "L", "BT", "V", "AR"]], data["HD"], test_size=0.3, random_state=10)
#gnb = SVC(kernel='linear')
#gnb = LinearSVC()
gnb =DecisionTreeClassifier()
#gnb= KNeighborsClassifier()



print("X_train value",X_train)
print("X_train length",len(X_train))
print("X_test length",len(X_test))

fit2 = gnb.fit(X_train, y_train)

predicer = fit2.predict(X_test)
print("Matrix:")
print(confusion_matrix(y_test, predicer))
print("Possibilty= ",accuracy_score(y_test,
        predicer))  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print("Score= ",accuracy_score(y_test, predicer, normalize=False))  # the number of correct predictions
print("len=",len(predicer))  #number of all of

