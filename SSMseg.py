from scipy.optimize import linear_sum_assignment
from scipy.linalg import norm
from math import atan
from math import sin, cos
from utils import *
from sklearn.decomposition import PCA
import pickle

def readImg(pathlist,h,w,mask=False):
    imglist=[]
    d=0 #depth that masks start to show
    for path in pathlist:
        try:
            img=cv2.imread(path)
            img=cv2.resize(img,(h,w))
            img=img.astype(np.float32)
            img/=255.0
        except:
            print('**exception** '+path)

        if mask:
            img=img[:,:,0] 
            img=img.reshape(h,w,1)
            img=toBinary(img,0.5)

                
        imglist.append(img)
    return imglist


def path2img(df_train,pred=True):

    imglist=[]
    masklist=[]
    predlist=[]
    for i in range(len(df_train['images'])):#img
        curr_list=readImg(df_train['images'][i],ImgHeight,ImgWidth)
        imglist.append(np.array(curr_list))
    for i in range(len(df_train['masks'])):#label
        curr_list=readImg(df_train['masks'][i],ImgHeight,ImgWidth,True)
        masklist.append(np.array(curr_list))
    if pred:
        print('size of raw image {}, mask {}, pred result {}'.format(len(df_train['images'][0]),len(df_train['masks'][0]),len(df_train['preds'][0])))
        for i in range(len(df_train['preds'])):#label
            curr_list=readImg(df_train['preds'][i],ImgHeight,ImgWidth,True)
            predlist.append(np.array(curr_list))
    return imglist,masklist,predlist

def get_shape(img):
    edges=cv2.Canny(np.uint8((img)),0,0)
    indices = np.where(edges != [0])
    yarr=np.array(indices[0])
    xarr=np.array(indices[1])
    #print('get_shape:the shape of y coordinates'+str(yarr.shape[0]))
    #print('get_shape:the shape of x coordinates'+str(xarr.shape[0]))
    return xarr,yarr


def toArr(x,y):
    
    arr=np.zeros((2*x.shape[0],))
    l=x.shape[0]
    for i in range(l):
        arr[2*i]=x[i]
        arr[2*i+1]=y[i]
    return arr
#we divide DEPTH into NUM levels, at every level, num of samplings is the minimum num among all 


def divideArr(arr,num_samples):
    #arr is original shape
    #return new shape 
    #x1,y1,x2,y2,x3,y3...
    length=arr.shape[0]/2
    step=(length-1)/(num_samples-1)
    new_arr=np.zeros((2*num_samples))
    new_arr[0]=arr[0]
    new_arr[1]=arr[1]
   
    for i in range(1,num_samples):
        idx=int(i*step)
        new_arr[2*i]=arr[2*idx]
        new_arr[2*i+1]=arr[2*idx+1]
    return new_arr


def get_translation(shape):
    mean_x = np.mean(shape[::2]).astype(np.int)
    mean_y = np.mean(shape[1::2]).astype(np.int)
    return np.array([mean_x, mean_y])

def translate(shape,h,w):
    mean_x, mean_y = get_translation(shape)
    shape=np.copy(shape)
    shape[::2] -= mean_x
    shape[::2]+=w//2
    shape[1::2] -= mean_y
    shape[1::2]+=h//2
    return shape
    
def get_rotation_scale(reference_shape, shape):
    
    a = np.dot(shape, reference_shape) / norm(reference_shape)**2
    
    #separate x and y for the sake of convenience
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]
    
    x = shape[::2]
    y = shape[1::2]
    
    b = np.sum(x*ref_y - ref_x*y) / norm(reference_shape)**2
    
    scale = np.sqrt(a**2+b**2)
    theta = atan(b / max(a, 10**-10)) #avoid dividing by 0
    
    return round(scale,1), round(theta,2)

def get_rotation_matrix(theta):
    
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def scale(shape, scale):
    
    return shape / scale

def rotate(shape, theta):
    
    matr = get_rotation_matrix(theta)
    
    #reshape so that dot product is eascily computed
    temp_shape = shape.reshape((-1,2)).T
    
    #rotate
    rotated_shape = np.dot(matr, temp_shape)
    
    return rotated_shape.T.reshape(-1)

def distance(ref_sh,sh):
    ref_x=ref_sh[::2]
    ref_y=ref_sh[1::2]
    x=sh[::2]
    y=sh[1::2]  
    return np.sum(np.sqrt((ref_x - x)**2 + (ref_y - y)**2))

def cost_matrix(shp,ref_shp):
    #shp 2*n , ref_shp 2*m , n>=m
    n=shp.shape[0]//2
    m=ref_shp.shape[0]//2
    assert n>=m,str(n)+' '+str(m)

    cost=np.zeros((n,n))
    #         \shp
    #  ref_shp\
    for i in range(m): #for each point in ref_shp
        for j in range(n):#for_each point in shp
            pt1=ref_shp[2*i:2*(i+1)]
            pt2=shp[2*j:2*(j+1)]
            dist=distance(pt1,pt2)
            cost[i][j]=dist
    return cost 
def correspond_and_cost(shp1,shp2):
    #shp1 > shp2
    cost=cost_matrix(shp1,shp2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind,col_ind,cost[row_ind, col_ind].sum()
#row is 0,1,2,3...

def correspondent_shape_contract(shp_r,shp_c,row,col):
    #row col is returned value after implementing hungarian algorithm 
    #return shp_c
    
    #choose m from n, 
    m=shp_r.shape[0]//2
    n=shp_c.shape[0]//2
    assert n>=m
    new_shp=np.zeros(shp_r.shape)
    #return new_shp, base on the shp_r
    
    #first fill 2*m
    for i in range(m):
        
        #new_shp[2*col[i]:2*(col[i]+1)]=shp_r[2*i:2*(i+1)]
        # move shp[col[i]] to shp[i]
        new_shp[2*i:2*(i+1)]=shp_c[2*col[i]:2*(col[i]+1)]
    return new_shp
    
    #expand m to n 
def correspondent_shape_expand(shp_r,shp_c,row,col):  
    #return shp_r
    m=shp_r.shape[0]//2
    n=shp_c.shape[0]//2
    assert n>=m
    new_shp=np.zeros(shp_c.shape)
    #i    move ref_shp[i] to ref_shp[col[i]]
    residual_shp=np.zeros((2*(n-m))) # of shp_c
    #base on shp_r
    for i in range(m):
        new_shp[2*col[i]:2*(col[i]+1)]=shp_r[2*i:2*(i+1)]
        
    if n>m:
        j=0
        for i in range(n):
            if  new_shp[2*i:2*(i+1)][0]==0 and new_shp[2*i:2*(i+1)][1]==0:
                residual_shp[2*j:2*(j+1)]=shp_c[2*i:2*(i+1)]
                j+=1
    #cost for residual shp and shp_r
    #n-m, m 
        if n-m<=m:
            row,col,cost=correspond_and_cost(shp_r,residual_shp)
            compli_shp_r=correspondent_shape_contract(residual_shp,shp_r,row,col)
#compli_shp_r is repeated points of shp_r to match the size of shp_c
        else:
            #n-m>m 
            row,col,cost=correspond_and_cost(residual_shp,shp_r)  
            compli_shp_r=correspondent_shape_expand(shp_r,residual_shp,row,col)
        j=0
        for i in range(n):
            if new_shp[2*i:2*(i+1)][0]==0 and new_shp[2*i:2*(i+1)][1]==0:
                new_shp[2*i:2*(i+1)]=compli_shp_r[2*j:2*(j+1)]
                j+=1 
        return new_shp
    
    else:
        return new_shp
        
def match_shape(ref_shp,shp): #return shp so that it has the same length as ref_shp
    #shp -> ref_shp
    #return new shp
    if ref_shp.shape[0]==0:
        return shp
    if shp.shape[0]==0:
        return 0
    
    if ref_shp.shape[0]>=shp.shape[0]:
        row,col,cost=correspond_and_cost(ref_shp,shp)
        return correspondent_shape_expand(shp,ref_shp,row,col)
    else:
        row,col,cost=correspond_and_cost(shp,ref_shp)
        return correspondent_shape_contract(ref_shp,shp,row,col)

def align_shape(reference_shape, shape): 

    #copy both shapes in caseoriginals are needed later
    temp_ref = np.copy(reference_shape)
    temp_sh = np.copy(shape)
 
    #translate(temp_ref)
    #translate(temp_sh)
    
    #get scale and rotation
    scale, theta = get_rotation_scale(temp_ref, temp_sh)
    
    #scale, rotate both shapes
    temp_sh = temp_sh / scale
    aligned_shape = rotate(temp_sh, theta)
    
    return aligned_shape

def restore(img,sh):
    
    assert sh.shape[0]>0
    x=sh[::2]
    y=sh[1::2]
    assert img.shape[0]>=np.amax(y) and img.shape[0]>=np.amax(x)
    for i in range(x.shape[0]):
        img[int(y[i])][int(x[i])]=1

def fill_edge(shape,h,w):
    #return mask of h and w
    img=np.zeros((h,w))
    if shape.shape[0]<=0:
        return img
    restore(img,shape)
   
    for i in range(w):
        #col? row[x][i] i is fixed x shifting
        row=img[i] #length is w
        up=0
        low=0
        for j in range(h):
            if img[j][i]!=0:
                up=j
                break
        for j in range(h-1,up,-1):#reverse from w-1
            if img[j][i]!=0:
                low=j
                break
        if up<low and img[up][i]!=0 and img[low][i]!=0:
           
            img[up+1:low,i]=1
    return img

def align(ref_sh,sh):
    if ref_sh.shape[0]<=0:
        return sh
    ref_sh=np.copy(ref_sh)
    sh=np.copy(sh)
    
    ref_x=ref_sh[::2]
    ref_y=ref_sh[1::2]
    x=sh[::2]
    y=sh[1::2]
    #return aligned shape 
    centr_x=np.mean(ref_x).astype(np.int)
    centr_y=np.mean(ref_y).astype(np.int)
    #move +centr_x-centr_other

    centr_xtmp=np.mean(x).astype(np.int)
    centr_ytmp=np.mean(y).astype(np.int)
    diffx=centr_xtmp-centr_x
    diffy=centr_ytmp-centr_y
        
    x-=diffx
    y-=diffy
    
    return toArr(x,y)
    
def img2shape(imglist):
    shapelist=[]
    max_samples=0  #when there are different points for aligned images , we either pick the min or max num of points
    min_samples=10000000
    minidx=0
    maxidx=0
    for img,idx in zip(imglist,range(len(imglist))):
        x,y=get_shape(img)
        arr=toArr(x,y)
        if x.shape[0]<=min_samples:
            min_samples=x.shape[0]
            minidx=idx
        if x.shape[0]>max_samples:
            max_samples=x.shape[0]
            maxidx=idx
        shapelist.append(arr)
    return shapelist,min_samples,max_samples,minidx,maxidx


def generalized_procrustes_analysis(imglist,do_expand=True):
    
    shapes,mins,maxs,minidx,maxidx=img2shape(imglist)
    #initialize Procrustes distance
    current_distance = 0
    
    if do_expand:
        tmp=shapes[0]
        shapes[0]=shapes[maxidx]
        shapes[maxidx]=tmp
    else:
        tmp=shapes[0]
        shapes[0]=shapes[minidx]
        shapes[minidx]=tmp        
    #initialize a mean shape
    mean_shape = shapes[0]

    num_shapes = len(shapes)
    
    #create array for new shapes, add 
    #new_shapes = np.zeros(np.array(shapes).shape)
    new_shapes=shapes

    while True:
        
        #add the mean shape as first element of array
        new_shapes[0] = mean_shape
        
        #superimpose all shapes to current mean
        for sh in range(1, num_shapes):

            new_sh = align(mean_shape, shapes[sh])#first move to the same center
            
            
            if do_expand:
                row,col,cost=correspond_and_cost(mean_shape,new_sh)
                final_sh=correspondent_shape_expand(new_sh,mean_shape,row,col)
            else:
                row,col,cost=correspond_and_cost(new_sh,mean_shape)
                final_sh=correspondent_shape_contract(mean_shape,new_sh,row,col)
            


            
            final_sh = align_shape(mean_shape, final_sh)
            new_shapes[sh] = final_sh
        
        #calculate new mean
        new_mean = np.mean(new_shapes, axis = 0)
        
        new_distance = distance(new_mean, mean_shape)
        #if the distance did not change, break the cycle
      
        if abs(new_distance-current_distance)<0.5:
            break
        #print('distance'+str(new_distance-current_distance))
        #align the new_mean to old mean
        
        new_mean=align(mean_shape,new_mean)
        new_mean = align_shape(mean_shape, new_mean)
        #update mean and distance
        mean_shape = new_mean
        current_distance = new_distance
    print("shape_size: "+str(mean_shape.shape[0]))
    print('--finish gpa\n')
    return mean_shape, new_shapes 

def PCA_analysis(shape,num_components):
    #shape is a np array [x,y,x,y,x,y...]
    pca=PCA(n_components=num_components)
    pca.fit(shape)
    return pca

def get_eigenvalues(pca):
    return pca.explained_variance_

def get_eigenvectors(pca):
    return pca.components_

def get_mean(pca):
    return pca.mean_

def storePickle(path,data):
    dbfile=open(path,'ab')
    pickle.dump(data,dbfile)
    dbfile.close()
    
def loadPickle(path):
    dbfile=open(path,'rb')
    db=pickle.load(dbfile)
    dbfile.close()
    return db

def divideDepth(depth,LEVEL):
    #equally divide depth into LEVEL range
    #0,...,depth-1
    step=(depth-1)/(LEVEL-1)
    ret=[]
    ret.append(0)
    for i in range(1,LEVEL):
        ret.append(int(i*step))
    return ret

def main():
    ImgHeight = 256
    ImgWidth =256
    pth_train=''
    pth_pickle=''
    pth_std='' 
    pth_shp=''
    pth_diff=''
    NUM_COMP=6

    df=load_into_df_3d(pth_train)
    images,masks,_=path2img(df,ImgHeight,ImgWidth,False)   
    masks,info=removeBlackImg_list(masks)

    for i in range(len(info)):
        images[i]=images[i][info[i][0]:info[i][1]+1]



    depthlist=[]
    for i in range(len(masks)):

        depthlist.append(masks[i].shape[0])
    LEVEL=int(np.mean(np.array(depthlist)))
    
    mean_z=[]
    mean_sh=[]
    shapelist=[]
    depth_arr=[] #[ Num [LEVEL],[],[]]
    for i in depthlist:
        tmp=divideDepth(i,LEVEL)
        depth_arr.append(tmp)
    depth_arr=np.array(depth_arr)

    for i in range(LEVEL): #Align all shapes 
        #calculate mean at every position 
        depths=depth_arr[:,i].reshape(-1)
        mean_z.append(np.mean(depths))
        msklist=[]
        for imgs,idx in zip(masks,depths):
            msklist.append(imgs[idx])
        sh,new_sh=generalized_procrustes_analysis(msklist)
        shapelist.append(new_sh)
        mean_sh.append(sh)


    pcalist=[]
    stdlist=[]
    difflist=[]
    fullDifflist=[]
    for i, shps in zip(range(LEVEL),shapelist):
    #weighted_sum+=weight(evalue[i],0.5)*evector[i]
        pca=PCA_analysis(shps,NUM_COMP)
        pcalist.append(pca)
        
        print(np.array(shps).shape)
        std=np.std(np.array(shps),axis=0)
        mean=get_mean(pca)
        diff=np.mean(shps-mean,axis=0)
        difflist.append(diff)
        stdlist.append(std)
        fullDifflist.append(shps-mean)
        
        evector=get_eigenvectors(pca)
        evalue=get_eigenvalues(pca)
        
        weights=np.repeat(evalue,(mean.shape[0])).reshape(evector.shape)
        shape=mean+np.sum(0.5*np.sqrt(weights)*evector,axis=0)
        img=np.zeros((256,256))
        restore(img,shape)
        showImg(img)
        print(i)
        i+=1
    #store generated ssm
    storePickle(pth_pickle,pcalist)
    storePickle(pth_std,stdlist)
    storePickle(pth_shp,shapelist)
    storePickle(pth_diff,difflist)

    print('testing variation')
    pca_test=pcalist[LEVEL//2]
    evector=get_eigenvectors(pca_test)
    evalue=get_eigenvalues(pca_test)
    mean=get_mean(pca_test)
    weights=np.repeat(evalue,(mean.shape[0])).reshape(evector.shape)

    for i in range(-3,4):
        shape=mean+np.sum(i*np.sqrt(weights)*evector,axis=0)
        img=np.zeros((256,256))
        restore(img,shape)
        showImg(img)


if __name__ == '__main__':
    main()