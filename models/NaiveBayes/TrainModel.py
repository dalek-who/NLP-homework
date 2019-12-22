import pandas as pd
df=pd.read_csv('G:\datafornlp\Train.csv').astype(str)  #读入训练集
df=df.iloc[:,0:5]
import jieba

Dict0= {}   #字典，用于存放出现在0类文本的所有单词，以及每个单词总共出现在多少个0类文本中
Dict1= {}   #字典，用于存放出现在1类文本的所有单词，以及每个单词总共出现在多少个1类文本中
Dict2= {}   #字典，用于存放出现在2类文本的所有单词，以及每个单词总共出现在多少个2类文本中

total0=0  #训练集中0类文本数量
total1=0  #训练集中1类文本数量
total2=0  #训练集中2类文本数量

for i in range(0,4868):
    tlist=(jieba.cut(df['title'][i]))
    clist=(jieba.cut(df['content'][i]))
    tp=int(df['label'][i])
    if (tp==0):
        total0+=1
    if (tp==1):
        total1+=1
    if (tp==2):
        total2+=1
    
    #提取当前文本中的单词，通过存放至集合set0，set1，set2去重    
    set0=set()   
    set1=set()
    set2=set()
    for j in tlist:
        if  (tp==0):
            set0.add(j)
        if  (tp==1):
            set1.add(j)
        if  (tp==2):
            set2.add(j)
    for j in clist:
        if  (tp==0):
            set0.add(j)
        if  (tp==1):
            set1.add(j)
        if  (tp==2):
            set2.add(j)
    #更新Dict0，Dict1，Dict2        
    for j in set0:
        if j in Dict0:
           Dict0[j]+=1
        else:
           Dict0[j]=1
    for j in set1:
        if j in Dict1:
           Dict1[j]+=1
        else:
           Dict1[j]=1
    for j in set2:
        if j in Dict2:
           Dict2[j]+=1
        else:
           Dict2[j]=1

#将Dict0，Dict1，Dict2分别存储至Type0.csv，Type1.csv，Type2.csv，用于调用模型时直接加载            
d0=pd.read_csv('G:\datafornlp\Type0.csv').astype(str)
d0=d0.iloc[:,0:2]
d1=pd.read_csv('G:\datafornlp\Type1.csv').astype(str)
d1=d1.iloc[:,0:2]
d2=pd.read_csv('G:\datafornlp\Type2.csv').astype(str)
d2=d2.iloc[:,0:2] 

for i in range(0,count0):
    dict0[d0['item'][i]]=d0['weight'][i]
for i in range(0,count2):
    dict2[d2['item'][i]]=d2['weight'][i]
for i in range(0,count1):
    dict1[d1['item'][i]]=d1['weight'][i]

    
