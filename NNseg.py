
from warnings import filterwarnings
filterwarnings('ignore')
from utils import *
from models import *


def name_gen(idx):
    l1=len(idx)
    lt=4-l1
    zero=''
    for i in range(lt):
        zero+='0'
    return 'axial'+zero+idx+'.png'

def load_into_df(DataPath):
    dirs=[]
    images = []
    masks = []
    for dirname, _, filenames in os.walk(DataPath):
        for filename in filenames:
            if '_mask'in filename :
                file_name=os.path.join(DataPath,dirname,filename)      
                dirs.append(dirname)
                masks.append(file_name)
                img_name=name_gen(filename[:-9])
                images.append(os.path.join(DataPath,dirname,img_name))
    return pd.DataFrame({'images': images, 'masks': masks})


def main():
    # specify Data path
    pth_train = './data/train'
    pth_test = './data/test'
    pth_pred= './data/train'
    BATCH_SIZE = 32
    ImgHeight = 256
    ImgWidth = 256
    BUFFER_SIZE = 1000

    # loading data
    df_train = load_into_df(pth_train)
    df_test = load_into_df(pth_test)
    img_new = []
    img_msk = []
    for i in range(len(df_train['images'])):
        if allBlack(df_train['masks'][i]):
            continue
        else:
            img_new.append(df_train['images'][i])
            img_msk.append(df_train['masks'][i])
    df_train_new = pd.DataFrame({'images': img_new, 'masks': img_msk})

    img_new_test = []
    msk_new_test = []
    for i in range(len(df_test['images'])):
        if allBlack(df_test['masks'][i]):
            continue
        else:
            img_new_test.append(df_test['images'][i])
            msk_new_test.append(df_test['masks'][i])
    df_test_new = pd.DataFrame({'images': img_new_test, 'masks': msk_new_test})

    trainds, testds = create_ds(df_train_new, df_test_new, BATCH_SIZE, BUFFER_SIZE)

    STEP_SIZE_TRAIN = len(df_train_new['images']) // BATCH_SIZE
    STEP_SIZE_VALID = len(df_test_new['images']) // BATCH_SIZE

    # specify your model
    model = unet2((ImgHeight, ImgWidth, 3))
    model.compile(optimizer='adam',
                  loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coef])
    model.summary()
    callbacks = [  # EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-3, verbose=1),
        ModelCheckpoint('Data_unet_2.h5', verbose=1, save_best_only=True, save_weights_only=True)]
    model.fit(trainds,
              steps_per_epoch=STEP_SIZE_TRAIN,
              epochs=60,
              callbacks=callbacks,
              validation_data=testds,
              validation_steps=STEP_SIZE_VALID)

    # storing labels of the model
    df_train = create_ds_2d(pth_pred)
    imglist = readImg(df_train['images'], ImgHeight, ImgWidth)
    store_pred_2d(model, imglist, df_train)

if __name__ == '__main__':
    main()