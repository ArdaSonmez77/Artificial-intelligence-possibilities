import csv
import matplotlib.pyplot as plt
import numpy
import pandas as pd

data = pd.read_csv("Dataset.csv")
#mt=float(data["R"])
#print(mt)
#print('max=',max(data["R"]))
#print('min=',min(data["R"]))
#print('Size=',len(data["R"]))

#R_arr = data["R"].tolist()

#rangee = len(R_arr) - 1

def variat(List):
    a=max(List)
    b=min(List)
    c=len(List)
    print(type(b))
    #rng=a-b
    for e in range(len(List)):

        z=(List[e]-b)/c
        print(float(z))
        #print(type(z))
        #break


print(data["R"])
List=[]
#List.to(data["R"])
max=0
min=1000
for i in range (len(List)):
    if List[i]<= min:
        min=List[i]
    elif List[i]>= max:
        max=List[i]

print(min)
print(max)
#variat(R_arr)

print(data)

print(data["ID"].describe())
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

