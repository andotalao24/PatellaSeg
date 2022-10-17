import cv2
import SimpleITK as sitk
import os
#The assumed file structure is as below
#./data
#   ./Mako 001
#       ./ axial.jpg


def mha2jpg(mhaPath,outFolder):

    #img = sitk.GetImageFromArray(nda)
    image = sitk.ReadImage(mhaPath)
    img_data = sitk.GetArrayFromImage(image)
    channel = img_data.shape[0]

    print(img_data.shape)
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    #max_num, min_num = np.max(img_data), np.min(img_data)
    #img_data = (img_data - min_num) / (max_num - min_num) * 255

    for s in range(channel):
        slicer = img_data[s,:,:]
        img = cv2.normalize(slicer, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # may need to change the name 
        cv2.imwrite(os.path.join(outFolder,str(s+1)+'_mask.jpg'),img)

def main():
    DataPath = ''
    msk_pth=''
    s=59
    e=103
    count=1
    for idx in range(s,e+1,1):
        if idx<10:
            name='Mako 00'+str(idx)
        elif idx<100:
            name='Mako 0'+str(idx)
        elif idx>=100:
            name='Mako '+str(idx)
        src=msk_pth.replace('Mako 001',name)
        out=DataPath.replace('Mako 001',name)
        try:
            mha2jpg(src,out)
            count+=1
        except:
            continue


if __name__ == '__main__':
    main()