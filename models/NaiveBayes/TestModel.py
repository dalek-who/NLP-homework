import pandas as pd
dg=pd.read_csv('G:\datafornlp\Test.csv').astype(str)
dg=df.iloc[:,0:5]



u=NaiveBayes()

anscorrect=0
answrong=0

for i in range(0,2027):
    m=3
    lab=int(dg['label'][i])
    m=u.classification(dg['title'][i]+dg['content'][i])
    
    if (m==lab):
        anscorrect+=1
    else:
        answrong+=1

print(anscorrect)  #anscorrect=1199
print(answrong)  #answrong=828

#在Test.csv的测试中，正确率1199/2027=0.59


m=u.classification('喜欢喜欢喜欢喜欢喜欢')  #结果为m=1
m=u.classification('我好喜欢你，因为你帅气又多金')  #结果为m=1
m=u.classification('他砸了我的厂子，真是罪大恶极') #结果为m=2
m=u.classification('改编不是乱编，戏说不是胡说') #结果为m=2
m=u.classification('贮藏充分') #结果为m=0