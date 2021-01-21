from data_gen import *
from model_pred import *
from sklearn.metrics import classification_report
import numpy as np
from tkinter import *
import os

# def session():
#     screen3.destroy()
#     global screen8
#     screen8=Tk()
#     screen8.title("Dashboard")
#     screen8.geometry("400x400")
#     Label(screen8,text="Welcome to the dashboard").pack()
#     Button(screen8,text="Logout",command=main_screen).pack()
#     Button(screen8, text="Start Detection",command=start_drowsiness_detection).pack()

# def login_success():
#     screen2.destroy()
#     global screen3
#     screen3=Tk()
#     screen3.title("Login Success!")
#     screen3.geometry("300x250")
#     Label(screen3,text="Login Success").pack()
#     Button(screen3, text="ok",command=session).pack()

# def user_not_found():
#     global screen5
#     screen5=Toplevel(screen)
#     screen5.title("User not recognized")
#     screen5.geometry("300x250")
#     Label(screen5,text="User not reconized").pack()
#     Button(screen5, text="ok",command=delete4).pack()

# def register_user():

#     username_info=username.get()
#     password_info=password.get()

#     username_entry.delete(0,END)
#     password_entry.delete(0,END)
#     line = str(username_info)+"___"+str(password_info)
#     modify_file(line)
#     Label(screen1, text="REgsiteration successful",fg="yellow",bg="red",font=("Calibri",11)).pack()

# def register():
#     screen.destroy()
#     global screen1
#     screen1=Tk()
#     screen1.title("Register")
#     screen1.geometry("300x250")

#     global username
#     global password
#     global username_entry
#     global password_entry

#     username=StringVar()
#     password=StringVar()

#     Label(screen1,text="Please enter details below*").pack()
#     Label(screen1,text=" ").pack()
#     Label(screen1,text="Username *").pack()

#     username_entry=Entry(screen1,textvariable=username)
#     username_entry.pack()

#     Label(screen1,text="Password *").pack()
#     password_entry=Entry(screen1,textvariable=password)
#     password_entry.pack()

#     Label(screen1,text="Password *").pack()
#     password_entry=Entry(screen1,textvariable=password)
#     Button(screen1,text="Register",width=10,height=1,bg="green",command=register_user).pack()

# def login_verify():
#     username1=username_verify.get()
#     password1=password_verify.get()
#     username_entry1.delete(0,END)
#     password_entry1.delete(0,END)

#     line = str(username1)+"___"+str(password1)+'\n'
#     user = open(r"Login_User.txt","r")
#     lines = user.readlines()
#     if line in lines:
#         login_success()
#     else:
#         user_not_found()
#     user.close()

# def login():
#     screen.destroy()
#     global screen2
#     screen2=Tk()
#     screen2.title("Login")
#     screen2.geometry("300x250")

#     Label(screen2,text="Please enter details below to login in").pack()
#     Label(screen2,text="").pack()

#     global username_verify
#     global password_verify

#     username_verify=StringVar()
#     password_verify=StringVar()

#     global username_entry1
#     global password_entry1

#     Label(screen2,text="Username *").pack()
#     username_entry1=Entry(screen2,textvariable=username_verify)
#     username_entry1.pack()
#     Label(screen2,text="Password").pack()
#     password_entry1=Entry(screen2,textvariable=password_verify)
#     password_entry1.pack()
#     Label(screen2,text="").pack()
#     Button(screen2,text="Login",width=10,height=1,command=login_verify).pack()

#     screen2.mainloop()

def main_screen():
    global screen
    screen=Tk()
    screen.geometry("300x250");
    screen.title("DDD");
    Label(text="DDD",bg="grey",width="300",height="2",font=("Calibri",13)).pack()
    Label(text="").pack()
    Button(text="Login",bg="yellow",height="2",width="30",command=login).pack()
    Label(text="").pack()
    Button(text="Register",bg="red",height="2",width="30",command=register).pack()
    screen.mainloop()

def start_drowsiness_detection():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    try:
        with open(r"Model\Eye_CNN_Model.json","r"):
            eye_model = load_eye_model()
    except FileNotFoundError:
        get_data("Eye")
        eye_model = load_eye_model()

    try:
        with open(r"Model\Yawn_CNN_Model.json","r"):
            yawn_model = load_yawn_model()
    except FileNotFoundError:
        get_data("Yawn")
        yawn_model = load_yawn_model()
    
    predict(eye_model, yawn_model)

def modify_file(line):
    user = open(r"Login_User.txt","a")
    user.write(line+"\n")
    user.close()   

if __name__ == "__main__":
    # g = Graphics()
    # main_screen()
    start_drowsiness_detection()