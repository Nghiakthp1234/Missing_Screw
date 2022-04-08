import tkinter as tk
from tkinter.ttk import *
import cv2 as cv
from os import path, listdir
import PIL.Image, PIL.ImageTk
from sub import DetectCircle
from keras.models import load_model
Image_id = []
Image_path = []
count = 0

#load model
model = load_model('model')
mode = 'opencv'

for filename in listdir("Honhai"):
    file_path = path.join("Honhai", filename)
    Image_path.append(file_path)
    Image_id.append(file_path[7:])
img = cv.imread('Honhai/2_10_15_11_50.jpg')

window = tk.Tk()
window.title("Detect missing screw by using opencv and deeplearning")

Main_canvas = tk.Canvas(window, width = 1255, height = 545, bd = 5, bg = 'purple')
Main_canvas.pack()

#Main_canvas.create_rectangle(10,10, 622,562, fill = 'white', width = 5, outline = 'white')
frame = cv.resize(img, (612,512))
frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))

btn_raw_img = tk.Button(window, image = photo, text = ('Raw Image'), font = ("Arial", 15),width = 612, height = 532,
                        compound = tk.TOP)
Main_canvas.create_window(10,10, anchor = tk.NW, window =btn_raw_img)

#img2 = cv.imread(r'D:\Nghia\Gui_missing_screw\Honhai\20200724_131833.jpg')

def update_img(img):
    global  btn_raw_img,photo
    frame = cv.resize(img, (612,512))
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))

    btn_raw_img.configure(image = photo)
update_img(img)

canvas_sub = tk.Canvas(window, width = 1255, height = 100)
canvas_sub.pack()

List_img = Combobox(window, width = 200, height = 20, textvariable = 'n', xscrollcommand = set())
List_img['values'] = Image_id
canvas_sub.create_window(1255,50, anchor = tk.SE, window = List_img, width = 400, height =30)

frame2 = cv.resize(img, (612,512))
frame2 = cv.cvtColor(frame2,cv.COLOR_BGR2RGB)
photo2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame2))
btn_export_img = tk.Button(window, image = photo2, text = ('Predicted Image'), font = ("Arial", 15),width = 612, height = 532,
                        compound = tk.TOP)
Main_canvas.create_window(642, 10, anchor = tk.NW, window = btn_export_img)

def Update_img_exp(img):
    global btn_export_img, photo2
    frame = cv.resize(img, (612, 512))
    frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
    btn_export_img.configure(image = photo2)
def Handle_btn():
    img_id = List_img.current()

    img = cv.imread(Image_path[img_id])
    update_img(img)
    #img = cv.imread(r'D:\Nghia\Gui_missing_screw\Honhai\2_11_10_2_28.jpg')

    img2 = DetectCircle(img,mode,model)
    Update_img_exp(img2)

    #img = cv.imread(r('D:\Nghia\Gui_missing_screw\Honhai\' +id))
button_chg_img = tk.Button(window, text = 'Confirm', command = Handle_btn, font = ("Arial", 20))
canvas_sub.create_window(700,20,anchor = tk.NW, window = button_chg_img)

mode_list = ['opencv', 'Dl model']
z = 0
#change_current_mode = mode_list[z]
lasted_mode = mode_list[z-1]
def change_mode():
    global  z, lasted_mode, button_change_mode, mode
    z+=-1
    if z== -1:
        z = 1
    mode = mode_list[z]
    lasted_mode = mode_list[z-1]
    #mode = mode_list[z]
    button_change_mode.configure(text = 'Current Mode: Detected missing screw by using ' + mode + '\n CLick to change to '+ lasted_mode +' mode')
button_change_mode = tk.Button(window,text = 'Current Mode: Detected missing screw by using ' + mode + '\n CLick to change to '+ lasted_mode +' mode', font= ("Arial", 18),
                               command = change_mode)
canvas_sub.create_window(0,10, anchor = tk.NW, window = button_change_mode)

window.mainloop()
