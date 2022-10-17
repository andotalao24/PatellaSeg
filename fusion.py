from SSMseg import *
from utils import *


class Point():
    def __init__(self, clustering, val):
        self.gp = clustering
        self.val = val

    def getClustering(self):
        return self.gp

    def getVal(self):
        return self.val

def dice_coef(y_true, y_pred,smooth=0.01):

    true_flat=np.reshape(y_true,[-1,])
    pred_flat=np.reshape(y_pred,[-1,])
    pred_flat[pred_flat>0.5]=1
    pred_flat[pred_flat<=0.5]=0
    intersection = 2 * np.sum(pred_flat * true_flat) + smooth
    denominator = np.sum(pred_flat) + np.sum(true_flat) + smooth
    return np.mean(intersection / denominator)

def get_test_shape(idx,mskid,masks,level):
    test_img=[masks[mskid][idx]]
    shapes,_,_,_,_=img2shape(test_img)
    test_shape=shapes[0]
    index=int(idx/masks[mskid].shape[0]*level)
    return test_shape,index,np.squeeze(np.array(test_img))

def average(pcalist,stdlist,index,pred_shape,x=5,z=2,y1=1): #VRS
    
    #pred_shape is input, which is the result from CNN
    pca=pcalist[index]
    mean=get_mean(pca)
    std=stdlist[index]
    mean=align(pred_shape,mean)
    if pred_shape.shape[0]<=0:
        return mean
    
    pred_shape=match_shape(mean,pred_shape)
    diff=abs(pred_shape-mean)

    a=1-diff/(x*(std))
    #when diff> x*std
    a[a>=1-z/x]=1 #diff too small <=z*std
    a[a<=0]=y1 # diff too big >=x*std

    ret=a*pred_shape+(1-a)*mean
    return ret

def get_score(pcalist,stdlist,index,pred_shape,a):
    pca=pcalist[index]
    mean=get_mean(pca)

    std=stdlist[index]
    mean=align(pred_shape,mean)
    pred_shape=match_shape(mean,pred_shape)

    avg_std=np.mean(std)
    devi_std=np.std(std)
    
    #d1,below avg_std
    #d2, between avg_std and avg_std+devi_std
    #d3 larger than avg_std+devi_std
    d1=[]
    d2=[]
    d3=[]
    
    for i in range(std.shape[0]//2):
        x=2*i
        y=x+1
        
        if std[x]+std[y]<=2*(avg_std-0.5*devi_std):
            d1.extend([x,y])
        elif std[x]+std[y]<=2*(avg_std+devi_std):
            d2.extend([x,y])
        else:
            d3.extend([x,y])

    img=np.zeros((256,256))
    restore(img,pred_shape[d1],10)
    restore(img,pred_shape[d2],50)
    restore(img,pred_shape[d3],200)


    D1=np.mean(abs(pred_shape[d1]-mean[d1]))
    D2=np.mean(abs(pred_shape[d2]-mean[d2]))
    D3=np.mean(abs(pred_shape[d3]-mean[d3]))
    
    return a*D3+(1-a)*D2,D1,a*D3+(1-a)*D2-D1

    
def score_test(index,pcalist,mask_img,test_shape,test_img,paras):
    '''
    Returns: difference score
    '''
    K=paras['k']
    w_A= paras['wa']
    b_A= paras['ba']
    w_B= paras['wb']
    b_B= paras['bb']

    smooth=0.01
    pca=pcalist[index]
    mean=get_mean(pca)
    mean=align(test_shape,mean)
    mean_img=fill_edge(mean,ImgHeight,ImgWidth)
    

    A=np.sum(test_img)
    B=np.sum(mean_img)

    I=np.sum(test_img*mean_img)
    Ae=np.sum(test_img-test_img*mean_img) #part "except" intersection of pred
    Be=np.sum(mean_img-test_img*mean_img)
    
    a=w_A/K*Ae+b_A
    b=-w_B/K*Be+b_B
    if a>1:
        a=1
    if b<0:
        b=0
    if I<=0:
        alpha1=0
    else:
        alpha1=np.sum(test_img*mean_img*mask_img)/I

    M=(A+B)/2
    s=((2*(alpha1*I+a*Ae)+smooth)/(A+M+smooth)-(2*(alpha1*I+b*Be)+smooth)/(B+M+smooth))

    return s
    
def regression(initial,masklist,predlist,pcalist,level):


    K=initial['k']
    M=initial['batch']
    w_A=initial['wa']
    b_A=initial['ba']
    w_B=initial['wb']
    b_B=initial['bb']
    lr=initial['lr']

    iteration=initial['iter']
    idx=1

    loss_a=np.zeros((M,))
    loss_b=np.zeros((M,))
    dw_A=np.zeros((M,))
    dw_B=np.zeros((M,))
    db_A=np.zeros((M,))
    db_B=np.zeros((M,))

    min_loss1=999
    min_loss2=999
    w_A_final=0
    b_A_final=0
    w_B_final=0
    b_B_final=0
    for n in range(iteration):
        print('iteration '+str(n))
        
        for i in range(len(masklist)):

            l=masklist[i].shape[0]
            
            for j in range(l):
                mask_shape,index,mask_img=get_test_shape(j,i,masklist,level)
                test_shape,index,test_img=get_test_shape(j,i,predlist,level)
                if(test_shape.shape[0]<=0):
                    continue
            
                mean=get_mean(pcalist[index])
                mean=align(test_shape,mean)
                mean_img=fill_edge(mean,ImgWidth,ImgHeight)

                Ae=np.sum(test_img-test_img*mean_img) 
                Be=np.sum(mean_img-test_img*mean_img)
            
                if Be<=0 or Ae<=0:
                    continue

                idx+=1
                true_a=np.sum((test_img-test_img*mean_img)*mask_img)/Ae
                true_b=np.sum((mean_img-test_img*mean_img)*mask_img)/Be
        
                y_a=w_A*(Ae/K)+b_A
                y_b=-w_B*(Be/K)+b_B
                
                loss_a[idx%M]=(true_a-y_a)**2
                loss_b[idx%M]=(true_b-y_b)**2
                
                dw_A[idx%M]=-2*(true_a-y_a)*(Ae/K)
                db_A[idx%M]=-2*(true_a-y_a)
                
                dw_B[idx%M]=2*(true_b-y_b)*(Be/K)
                db_B[idx%M]=-2*(true_b-y_b)
                    
                if idx%M==0:
                    la=np.mean(loss_a)
                    lb=np.mean(loss_b)
                    if la<min_loss1:
                        w_A_final=w_A
                        b_A_final=b_A
                        min_loss1=la
                    if lb<min_loss2:
                        w_B_final=w_B
                        b_B_final=b_B
                        min_loss2=lb
                    
                    #updating
                    w_A=w_A-np.mean(dw_A)*lr
                    w_B=w_B-np.mean(dw_B)*lr
                    b_A=b_A-np.mean(db_A)*lr
                    b_B=b_B-np.mean(db_B)*lr
            
                    print('loss_a '+str(la))
                    print('loss_b '+str(lb))

    return w_A_final,b_A_final,w_B_final,b_B_final



def rm_outlier_stats(an_array,n=2):
    #larger than n *dev considered as outliers
    #return dictionary
    an_array=np.array(an_array)
    mean = np.mean(an_array)
    standard_deviation = np.std(an_array)
    distance_from_mean = abs(an_array - mean)
    max_deviations = n
    not_outlier = distance_from_mean <= max_deviations * standard_deviation
    return {'mean':np.mean(an_array[not_outlier]),'std':np.std(an_array[not_outlier])}


def clustering(masklist,predlist,pcalist,stdlist,level,paras):
    ssm_k = []
    cnn_k = []
    avg_k = []

    for i in range(len(masklist)):  # i is different sample
        avg_dice_process = []
        avg_diff_local = []
        l = masklist[i].shape[0]

        for j in range(l):  # j is index
            mask_shape, index, mask_img = get_test_shape(j, i, masklist, level)
            test_shape, index, test_img = get_test_shape(j, i, predlist, level)
            if test_shape.shape[0] <= 0:
                continue

            mean = get_mean(pcalist[index])
            mean = align(test_shape, mean)
            mean_img = fill_edge(mean, 256, 256)
            # test_img=fill_edge(test_shape,256,256)
            d1 = dice_coef(mask_img, test_img)
            d2 = dice_coef(mask_img, mean_img)
            score= score_test(index, pcalist, mask_img, test_shape,test_img,paras)

            # score nan choose ssm
            if math.isnan(score):
                score = 0
            if not math.isnan(score):
                avg_diff_local.append((100 * (d1 - d2) - 100 * score) ** 2)
            processed_shp = average(pcalist, stdlist, index, test_shape)
            new_img = fill_edge(processed_shp, 256, 256)
            d3 = dice_coef(mask_img, new_img)

            if d1 >= d2 and d1 >= d3:
                cnn_k.append(score)
            elif d2 >= d1 and d2 >= d3:
                ssm_k.append(score)
            elif d3 >= d1 and d3 >= d2:
                avg_k.append(score)

        if len(avg_dice_process) == 0:
            continue
    ssm_stats = rm_outlier_stats(ssm_k)
    cnn_stats = rm_outlier_stats(cnn_k)
    avg_stats = rm_outlier_stats(avg_k)
    #by default, there are k=3 points in one cluster
    ssm_ = [ssm_stats['mean'], ssm_stats['mean'] + 0.5 * ssm_stats['std'], ssm_stats['mean'] - 0.5 * ssm_stats['std']]
    cnn_ = [cnn_stats['mean'], cnn_stats['mean'] + 0.5 * cnn_stats['std'], cnn_stats['mean'] - 0.5 * cnn_stats['std']]
    avg_ = [avg_stats['mean'], avg_stats['mean'] + 0.5 * avg_stats['std'], avg_stats['mean'] - 0.5 * avg_stats['std']]

    return [cnn_, ssm_, avg_]


def compare(p1, p2):
    return p1.getVal() < p2.getVal()


def fuzzy_knn(k, score, lists):
    # lists: [cnn:[scores],ssm:[scores],avg:[scores] ]
    # stats;[cnn_dic:{mean:,std:},ssm_dic, avg_dic]
    # score:an int,
    # return weights for cnn ssm avg respectively
    # str 'cnn', 'ssm', 'avg'
    # 0 cnn, 1 ssm, 2 avg
    def dis(a, b):
        return abs(a - b)

    def dis2(a, b):
        smooth = 0.0001
        return 1 / ((a - b) ** 2 + smooth)

    def weight(score, k_pts, typ):
        # cnn
        up = 0
        down = 0
        for pt in k_pts:
            if pt.getClustering() == typ:
                if score == pt.getVal():
                    return 1
                else:
                    up += dis2(score, pt.getVal())
            down += dis2(score, pt.getVal())
        return up / down

    dic = dict()  # cnn:[n1,n2,...nk],smallest k distances
    # k num ascending from 0
    for n in range(3):
        min_k = []
        for i in range(len(lists[n])):
            d = dis(lists[n][i], score)
            min_k.append(d)
            for j in range(len(min_k) - 1, 0, -1):  # from n-1 to 1
                if min_k[j] < min_k[j - 1]:
                    tmp = min_k[j - 1]
                    min_k[j - 1] = min_k[j]
                    min_k[j] = tmp
            if len(min_k) > k:
                min_k = min_k[:-1]

        dic[n] = min_k

    k_pts = []  # to store the k nearest points
    for key, val_list in dic.items():
        for val in val_list:
            pt = Point(key, val)
            k_pts.append(pt)
            for j in range(len(k_pts) - 1, 0, -1):
                if compare(k_pts[j], k_pts[j - 1]):
                    tmp = k_pts[j - 1]
                    k_pts[j - 1] = k_pts[j]
                    k_pts[j] = tmp
            if len(k_pts) > k:
                k_pts = k_pts[:-1]

    w_cnn = weight(score, k_pts, 0)
    w_ssm = weight(score, k_pts, 1)
    w_avg = weight(score, k_pts, 2)

    return w_cnn, w_ssm, w_avg

def eval(scorelist,predlist,pcalist,stdlist,masklist,level,paras,k=3):

    for i in range(len(masklist)):  # i is different sample
        avg_dice_process = []
        l = masklist[i].shape[0]
        mean_arr = np.zeros(masklist[i].shape)

        for j in range(l):  # j is index
            mask_shape, index, mask_img = get_test_shape(j, i, masklist,level)
            test_shape, index, test_img = get_test_shape(j, i, predlist,level)
            mean = get_mean(pcalist[index])
            mean = align(test_shape, mean)
            mean_img = fill_edge(mean, 256, 256)
            mean_arr[j] = mean_img
            d1 = dice_coef(mask_img, test_img)  # dsc cnn
            d2 = dice_coef(mask_img, mean_img)  # dsc ssm
            score = score_test(index, pcalist, mask_img, test_shape,test_img,paras)

            #scorelist:[cnn, ssm, avg]
            w1, w2, w3 = fuzzy_knn(k, score, scorelist)

            if math.isnan(score):
                w1 = 1
                w2 = 0
                w3 = 0
            processed_shp = average(pcalist, stdlist, index, test_shape)
            img = fill_edge(processed_shp, 256, 256)
            new_img = (w1 * test_img + w2 * mean_img + w3 * img)
            d3 = dice_coef(mask_img, new_img)

            print('{} case, {} slice, ssm: {}, cnn: {}, combined: {}'.format(i, j, d2, d1, d3))

            avg_dice_process.append(d3)
        if len(avg_dice_process) == 0:
            continue
        print('avg_dice with processed ' + str(np.mean(np.array(avg_dice_process))))

def main():

    pth_test=''
    pth_val=''
    pth_pickle=''
    pth_std=''

    pcalist=loadPickle(pth_pickle)
    LEVEL=len(pcalist)
    stdlist=loadPickle(pth_std)

    df=load_into_df_3d_pred(pth_test)
    imglist,masklist,predlist=path2img(df,True)
    masklist,info=removeBlackImg_list(masklist)

    for i in range(len(info)):
        predlist[i]=predlist[i][info[i][0]:info[i][1]+1]
        imglist[i]=imglist[i][info[i][0]:info[i][1]+1]

    df_val=load_into_df_3d_pred(pth_val)
    imglist_val,masklist_val,predlist_val=path2img(df_val,True)
    masklist_val,info=removeBlackImg_list(masklist_val)

    for i in range(len(info)):
        predlist_val[i]=predlist_val[i][info[i][0]:info[i][1]+1]
        imglist_val[i]=imglist_val[i][info[i][0]:info[i][1]+1]
        try:
            if len(imglist_val[i])!=len(predlist_val[i]):
                print(len(imglist_val[i]),len(predlist_val[i]))
        except:
            break
    #k is a hyperparameter to scale the size of Se and Ce. see the paper for details
    initial={'k':200,'batch':32,'wa':1,'ba':0.5,'wb':1,'bb':0.5,'lr':0.1,'iter':10}
    w_A_final,b_A_final,w_B_final,b_B_final=regression(initial,masklist_val,predlist_val,pcalist,LEVEL)
    paras={'k':200,'wa':w_A_final,'ba':b_A_final,'wb':w_B_final,'bb':b_B_final}
    scorelist=clustering(masklist_val,predlist_val,pcalist,stdlist,LEVEL,paras)
    eval(scorelist,predlist,pcalist,stdlist,masklist,LEVEL,paras)

if __name__ == '__main__':
    main()