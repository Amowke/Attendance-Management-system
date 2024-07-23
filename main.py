from tkinter import*
from customtkinter import *
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image,ImageTk
import os
import mysql.connector
from time import strftime
from datetime import datetime

set_appearance_mode("light")
set_default_color_theme("blue")
app = CTk()
app.title("Attendance Management System")
app.geometry("1000x900+0+0")



# Function to open students.py
def open_students_details():
    os.system('python students.py')

# Function to open photos folder
def open_img():
    os.startfile("data")

# Function to open attendance.py
def open_attendance():
    os.system("python attendance.py")    

#train data function
def train_classifier():
    data_dir = ("data")
    path = [os.path.join(data_dir,file) for file in os.listdir(data_dir)]
    
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img,'uint8')
        id = int(os.path.split(image)[1].split('.')[1])
 
        faces.append(imageNp)
        ids.append(id)
        cv2.imshow("Training", imageNp)
        cv2.waitKey(1)==13

    ids = np.array(ids)

    # Train the classifier and save the model
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")
    cv2.destroyAllWindows()
    messagebox.showinfo("Result", "Training dataset completed!!")

#========== attendance ==========
def mark_attendance(i,r,n,d):  
    with open("attendance.csv","r+",newline="\n") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split((","))
            nameList.append(entry[0])
        if ((i not in nameList) and (r not in nameList) and (n not in nameList) and (d not in nameList)):
            now = datetime.now()
            d1 = now.strftime("%d/%m/%Y")
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{i},{r},{n},{d},{dtString},{d1},Present")  


#======= face recognition =========
def face_recog():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        coord = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
            id,predict = clf.predict(gray_image[y:y+h, x:x+w])
            confidence = int((100 * (1 - predict/300)))

            conn = mysql.connector.connect(
                host="localhost",
                username="root",
                password="granted",
                database="face_recognizer"
            )
            my_cursor = conn.cursor()
            
            my_cursor.execute("select Name from student where Student_id="+str(id))
            n = my_cursor.fetchone()
            n="+".join(n)

            my_cursor.execute("select Roll from student where Student_id="+str(id))
            r = my_cursor.fetchone()
            r="+".join(r)

            my_cursor.execute("select Dep from student where Student_id="+str(id))
            d = my_cursor.fetchone()
            d="+".join(d)

            my_cursor.execute("select Student_id from student where Student_id="+str(id))
            i = my_cursor.fetchone()
            i="+".join(i)


        
            if confidence > 77:
                cv2.putText(img, f"ID:{i}", (x, y-75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 3)
                cv2.putText(img, f"Roll:{r}", (x, y-55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 3)
                cv2.putText(img, f"Name:{n}", (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 3)
                cv2.putText(img, f"Department:{d}", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 3)
                mark_attendance(i,r,n,d)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(img, "Unknown Face", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 3)

            coord = [x, y, w, y]

            conn.close()

        return coord

    def recognize(img, clf, faceCascade):
        coord = draw_boundary(img, faceCascade, 1.1, 10, (255, 25, 255), "Face", clf)
        return img
    
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_cap = cv2.VideoCapture(0)

    while True:
        ret, img = video_cap.read()
        img = recognize(img, clf, faceCascade)
        cv2.imshow("face detection", img)

        if cv2.waitKey(1) == 13:
            break
    video_cap.release()
    cv2.destroyAllWindows()



#sidebar
frame1 = CTkFrame(app, width=400)
frame1.pack(fill="y", side="left")

lbl = CTkLabel(app,text="Click any button to view the content",font=("Arial", 20))
lbl.pack(padx=200, pady=400, anchor="center")



# student button   
btn_student=CTkButton(frame1,text="Student Details",cursor="hand2",font=("times new roman",15,"bold"),width=220,height=40, command=open_students_details)
btn_student.grid(row=0,column=0,padx=10,pady=10)


# Face Detector button
b1_2=CTkButton(frame1,text="Face Detector",cursor="hand2",command=face_recog,font=("times new roman",15,"bold"),width=220,height=40)
b1_2.grid(row=1,column=0,padx=10,pady=10)


# Attendance button
b1_3=CTkButton(frame1,text="Attendance",cursor="hand2",command=open_attendance,font=("times new roman",15,"bold"),width=220,height=40)
b1_3.grid(row=2,column=0,padx=10,pady=10)


# Train Data button
b1_4=CTkButton(frame1,text="Train Data",command=train_classifier,cursor="hand2",font=("times new roman",15,"bold"),width=220,height=40)
b1_4.grid(row=3,column=0,padx=10,pady=10)


# Photos button
btn_photos=CTkButton(frame1,text="Photos",cursor="hand2",command=open_img,font=("times new roman",15,"bold"),width=220,height=40)
btn_photos.grid(row=4,column=0,padx=10,pady=10)


# Function to show exit confirmation dialog
def on_exit_clicked():
    response = messagebox.askyesno("Exit","Are you sure you want to exit?")
    if response:
        app.quit()

# Exit button  
btn_exit=CTkButton(frame1,text="Exit",cursor="hand2",font=("times new roman",15,"bold"),width=220,height=40,command=on_exit_clicked)
btn_exit.grid(row=6,column=0,padx=10,pady=10)









app.mainloop()