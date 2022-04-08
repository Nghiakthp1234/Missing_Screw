import numpy as np

#Load lại model đã train
#model = keras.models.load_model(r'D:\Nghia\Gui_missing_screw\model')
def DL_Check_Missing_Screw(img,model):
    img = np.array(img)
    #reshape về giá trị đầu vào model
    img = np.reshape(img, (1,80,80,3))

    #tiến hành predict
    values = model.predict(img)
    values = np.where(values[0] == 1)
    return values[0][0]
