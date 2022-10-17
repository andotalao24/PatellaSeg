import os
import random
import shutil
#Split all data into train, validation and test

def walkAllFiles(path,te,val,tr,lastTe=[]):
    caseList=[]
    tmpl=[]
    for (rt,dirs,files) in os.walk(path):
        tmpl=dirs
        break

    n=len(tmpl)
    assert te+val+tr==n
    for i in range(n):
        caseList.append((i,os.path.join(path,tmpl[i]),tmpl[i]))

    random.shuffle(caseList)
    if len(lastTe)>0:#avoid choosing the same file for testing as ones of the last test data
        #def cmp(e):
            #return e[0]

        #lastTe.sort(key=cmp)
        idxlist=[it[0] for it in lastTe]
        k=0
        for i in range(n):
            if caseList[i][0] not in idxlist:
                tmp=caseList[k]
                caseList[k]=caseList[i]
                caseList[i]=tmp
                k+=1
     
    test=caseList[:te]
    validation=caseList[te:te+val]
    train=caseList[-tr:]

    return test,train,validation

def moveFile(list,path,name):
    ppath=os.path.join(path,name)
    os.mkdir(ppath)
    for case in list:
        shutil.move(case[1],os.path.join(ppath,case[2]))


def main():
    te=15
    val=20
    tr=50
    path=r'.\split1'
    path2=r'.\split2'
    test,train,validation=walkAllFiles(path,te,val,tr)
    moveFile(test,path,'test')
    moveFile(train,path,'train')
    moveFile(validation,path,'valid')

    test,train,validation=walkAllFiles(path2,te,val,tr,test)
    moveFile(test,path2,'test')
    moveFile(train,path2,'train')
    moveFile(validation,path2,'valid')

if __name__=='__main__':
    main()
