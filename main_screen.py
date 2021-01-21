from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import tkinter
import os
import random
import cv2

live_video = cv2.VideoCapture(0)
def quit_win():
    bl = messagebox.askquestion("Question", "Do you want to quit")
    if (bl == 'yes'):
        window.quit()


window = tkinter.Tk()
window.geometry('1280x720')
window.resizable(1, 1)
window.title('Driver Drowsiness Detection System')


## Create a Labels

name_label = ttk.Label(text='Name :')
age_label = ttk.Label(text='Age :')
number_label = ttk.Label(text='Phone Number :')
mail_label = ttk.Label(text='Email id :')

val_x = 0
val_y = 5

name_label.grid(row=0, column=1, sticky=W, pady=val_y, padx=val_x)
age_label.grid(row=1, column=1, sticky=W, pady=val_y, padx=val_x)
number_label.grid(row=4, column=1, sticky=W, pady=val_y, padx=val_x)
mail_label.grid(row=5, column=1, sticky=W, pady=val_y, padx=val_x)

## Create a Entry Fields And Radio Button For Form

name_entry = Entry()
age_entry = Entry()


number_entry = Entry()
mail_entry = Entry()

val_x = 3
val_y = 3

name_entry.grid(row=0, column=2, padx=(2, 15), ipadx=val_x, ipady=val_y)
age_entry.grid(row=1, column=2, padx=(2, 15), ipadx=val_x, ipady=val_y)

number_entry.grid(row=4, column=2, padx=(2, 15), ipadx=val_x, ipady=val_y)
mail_entry.grid(row=5, column=2, padx=(2, 15), ipadx=val_x, ipady=val_y)

## Create Buttons

exit = ttk.Button(text='Exit', command=quit_win, )
exit.grid(row=7, column=1, columnspan=2, sticky=W + E, padx=(2, 15), pady=(0, 15), ipadx=3, ipady=3)

# Add a Image



img_lb = Label(image=image)
img_lb.image = image
img_lb.grid(row=0, column=0, rowspan=6)

window.mainloop()
