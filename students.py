from tkinter import *
from customtkinter import *
from tkinter import ttk
from tkinter import messagebox
import mysql.connector
import cv2

set_appearance_mode("light")
set_default_color_theme("blue")
app = CTk()
app.title("Attendance Management System")
app.geometry("1900x700+0+0")

# =========== function declaration ==========

def add_data():
    if app.var_dep.get() == "Select Department" or app.var_std_name.get() == "" or app.var_std_id.get() == "":
        messagebox.showerror("Error", "All fields are required", parent=app)
    else:
        try:
            conn = mysql.connector.connect(host="localhost", username="root", password="granted", database="face_recognizer")
            my_cursor = conn.cursor()
            my_cursor.execute("insert into student values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (
                app.var_dep.get(),
                app.var_course.get(),
                app.var_year.get(),
                app.var_semester.get(),
                app.var_std_id.get(),
                app.var_std_name.get(),
                app.var_div.get(),
                app.var_roll.get(),
                app.var_gender.get(),
                app.var_dob.get(),
                app.var_email.get(),
                app.var_phone.get(),
                app.var_address.get(),
                app.var_teacher.get(),
                app.radiovar.get()
            ))
            conn.commit()
            fetch_data()
            conn.close()
            messagebox.showinfo("Success", "Student details have been added successfully", parent=app)
        except Exception as es:
            messagebox.showerror("Error", f"Due To :{str(es)}", parent=app)

#========= fetch data ==========
def fetch_data():
    conn = mysql.connector.connect(host="localhost", username="root", password="granted", database="face_recognizer")
    my_cursor = conn.cursor()
    my_cursor.execute("select * from student")
    data = my_cursor.fetchall()

    if len(data) != 0:
        app.student_table.delete(*app.student_table.get_children())
        for i in data:
            app.student_table.insert("", END, values=i)
        conn.commit()
    conn.close()

#======== get cursor ==========
def get_cursor(event=""):
    cursor_focus = app.student_table.focus()
    content = app.student_table.item(cursor_focus)
    data = content["values"]

    app.var_dep.set(data[0])
    app.var_course.set(data[1])
    app.var_year.set(data[2])
    app.var_semester.set(data[3])
    app.var_std_id.set(data[4])
    app.var_std_name.set(data[5])
    app.var_div.set(data[6])
    app.var_roll.set(data[7])
    app.var_gender.set(data[8])
    app.var_dob.set(data[9])
    app.var_email.set(data[10])
    app.var_phone.set(data[11])
    app.var_address.set(data[12])
    app.var_teacher.set(data[13])
    app.radiovar.set(data[14])

# ========= update function ==========
def update_data():
    if app.var_dep.get() == "Select Department" or app.var_std_name.get() == "" or app.var_std_id.get() == "":
        messagebox.showerror("Error", "All fields are required", parent=app)
    else:
        try:
            Update = messagebox.askyesno("Update", "Do you want to update this student's details", parent=app)
            if Update > 0:
                conn = mysql.connector.connect(host="localhost", username="root", password="granted", database="face_recognizer")
                my_cursor = conn.cursor()
                my_cursor.execute("update student set Dep=%s, Course=%s, Year=%s, Semester=%s, Name=%s, Division=%s, Roll=%s, Gender=%s, Dob=%s, Email=%s, Phone=%s, Address=%s, Teacher=%s, PhotoSample=%s where Student_id=%s", (
                    app.var_dep.get(),
                    app.var_course.get(),
                    app.var_year.get(),
                    app.var_semester.get(),
                    app.var_std_name.get(),
                    app.var_div.get(),
                    app.var_roll.get(),
                    app.var_gender.get(),
                    app.var_dob.get(),
                    app.var_email.get(),
                    app.var_phone.get(),
                    app.var_address.get(),
                    app.var_teacher.get(),
                    app.radiovar.get(),
                    app.var_std_id.get()
                ))
            else:
                if not Update:
                    return
            messagebox.showinfo("Success", "Student details have been updated successfully", parent=app)
            conn.commit()
            fetch_data()
            conn.close()
        except Exception as es:
            messagebox.showerror("Error", f"Due To :{str(es)}", parent=app)

#======== delete function ==========
def delete_data():
    if app.var_std_id.get() == "":
        messagebox.showerror("Error", "Student ID must be required", parent=app)
    else:
        try:
            delete = messagebox.askyesno("Student Delete Page", "Do you want to delete this student", parent=app)
            if delete > 0:
                conn = mysql.connector.connect(host="localhost", username="root", password="granted", database="face_recognizer")
                my_cursor = conn.cursor()
                sql = "delete from student where Student_id=%s"
                val = (app.var_std_id.get(),)
                my_cursor.execute(sql, val)
            else:
                if not delete:
                    return
            conn.commit()
            fetch_data()
            conn.close()
            messagebox.showinfo("Delete", "Student details have been deleted successfully", parent=app)
        except Exception as es:
            messagebox.showerror("Error", f"Due To :{str(es)}", parent=app)

#======== reset function ==========
def reset_data():
    app.var_dep.set("Select Department")
    app.var_course.set("Select Course")
    app.var_year.set("Select Year")
    app.var_semester.set("Select Semester")
    app.var_std_id.set("")
    app.var_std_name.set("")
    app.var_div.set("Select")
    app.var_roll.set("")
    app.var_gender.set("Select")
    app.var_dob.set("")
    app.var_email.set("")
    app.var_phone.set("")
    app.var_address.set("")
    app.var_teacher.set("")
    app.radiovar.set("")

# ========= generate dataset or take photo sample ==========
def generate_dataset():
    if app.var_dep.get() == "Select Department" or app.var_std_name.get() == "" or app.var_std_id.get() == "":
        messagebox.showerror("Error", "All fields are required", parent=app)
    else:
        try:
            conn = mysql.connector.connect(host="localhost", username="root", password="granted", database="face_recognizer")
            my_cursor = conn.cursor()
            my_cursor.execute("select * from student")
            myresult = my_cursor.fetchall()
            id = 0
            for x in myresult:
                id += 1
            my_cursor.execute("update student set Dep=%s, Course=%s, Year=%s, Semester=%s, Name=%s, Division=%s, Roll=%s, Gender=%s, Dob=%s, Email=%s, Phone=%s, Address=%s, Teacher=%s, PhotoSample=%s where Student_id=%s", (
                app.var_dep.get(),
                app.var_course.get(),
                app.var_year.get(),
                app.var_semester.get(),
                app.var_std_name.get(),
                app.var_div.get(),
                app.var_roll.get(),
                app.var_gender.get(),
                app.var_dob.get(),
                app.var_email.get(),
                app.var_phone.get(),
                app.var_address.get(),
                app.var_teacher.get(),
                app.radiovar.get(),
                id + 1
            ))
            conn.commit()
            fetch_data()
            reset_data()
            conn.close()

            # Load prebuilt model for face detection
            face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

            def face_cropped(img):
                faces = face_classifier.detectMultiScale(img, 1.3, 5)
                # scaling factor = 1.3
                # Minimum neighbor = 5
                for (x, y, w, h) in faces:
                    face_cropped = img[y:y+h, x:x+w]
                    return face_cropped

            cap = cv2.VideoCapture(0)
            img_id = 0
            while True:
                ret, my_frame = cap.read()
                if face_cropped(my_frame) is not None:
                    img_id += 1
                    face = cv2.resize(face_cropped(my_frame), (450, 450))
                    file_name_path = "data/user." + str(id) + "." + str(img_id) + ".jpg"
                    cv2.imwrite(file_name_path, face)
                    cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    cv2.imshow("Cropped Face", face)

                if cv2.waitKey(1) == 13 or int(img_id) == 100:
                    break
            cap.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("Result", "Generating dataset completed!!!")
        except Exception as es:
            messagebox.showerror("Error", f"Due To :{str(es)}", parent=app)

# ========= update photo sample ==========
def update_photo_sample():
    if app.var_std_id.get() == "":
        messagebox.showerror("Error", "Student ID must be required", parent=app)
    else:
        try:
            conn = mysql.connector.connect(host="localhost", username="root", password="granted", database="face_recognizer")
            my_cursor = conn.cursor()
            my_cursor.execute("select * from student where Student_id=%s", (app.var_std_id.get(),))
            myresult = my_cursor.fetchall()
            id = app.var_std_id.get()
            conn.close()

            # Load prebuilt model for face detection
            face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

            def face_cropped(img):
                faces = face_classifier.detectMultiScale(img, 1.3, 5)
                for (x, y, w, h) in faces:
                    face_cropped = img[y:y+h, x:x+w]
                    return face_cropped

            cap = cv2.VideoCapture(0)
            img_id = 0
            while True:
                ret, my_frame = cap.read()
                if face_cropped(my_frame) is not None:
                    img_id += 1
                    face = cv2.resize(face_cropped(my_frame), (450, 450))
                    file_name_path = "data/user." + str(id) + "." + str(img_id) + ".jpg"
                    cv2.imwrite(file_name_path, face)
                    cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    cv2.imshow("Cropped Face", face)

                if cv2.waitKey(1) == 13 or int(img_id) == 100:
                    break
            cap.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("Result", "Updating photo sample completed!!!")
        except Exception as es:
            messagebox.showerror("Error", f"Due To :{str(es)}", parent=app)

            

# ========= variables ==========
app.var_dep = StringVar()
app.var_course = StringVar()
app.var_year = StringVar()
app.var_semester = StringVar()
app.var_std_id = StringVar()
app.var_std_name = StringVar()
app.var_div = StringVar()
app.var_roll = StringVar()
app.var_gender = StringVar()
app.var_dob = StringVar()
app.var_email = StringVar()
app.var_phone = StringVar()
app.var_address = StringVar()
app.var_teacher = StringVar()

# Left label
left_lbl = CTkLabel(app, text="Student Details", font=("Arial", 20, "bold"), text_color="black")
left_lbl.place(x=100, y=99)

# Left Frame
left_frame = CTkFrame(app, border_width=2, border_color="black", width=800, height=520, fg_color="white")
left_frame.place(x=100, y=130)
left_frame.grid_propagate(False)

# current course
current_course_lbl = CTkLabel(left_frame, text="Current Course", font=("Arial", 15, "bold"), text_color="black")
current_course_lbl.grid(row=0, column=0, padx=10, pady=5, sticky=W)

current_course_frame = CTkFrame(left_frame, fg_color="white", border_width=2, border_color="black", width=600, height=100)
current_course_frame.grid(row=1, column=0, padx=10, sticky=W)
current_course_frame.grid_propagate(False)

# Department
dep_lbl = CTkLabel(current_course_frame, text="Department", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
dep_lbl.grid(row=0, column=0, padx=10, sticky=W)

departments = ["Select Department", "Computer Science", "Mathematics", "Actuarial Science", "Physics", "Biochemistry"]
dep_combo = CTkComboBox(current_course_frame, values=departments, variable=app.var_dep)
dep_combo.set("Select Department")
dep_combo.grid(row=0, column=1, padx=2, pady=10, sticky=W)

# course
course_lbl = CTkLabel(current_course_frame, text="Course", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
course_lbl.grid(row=0, column=2, padx=10, sticky=W)

courses = ["Select Course", "Database", "Data Structures", "F.A", "C.A", "Java", "Electronics", "Pure Maths"]
course_combo = CTkComboBox(current_course_frame, values=courses, variable=app.var_course)
course_combo.set("Select Course")
course_combo.grid(row=0, column=3, padx=2, pady=10, sticky=W)

# Year
year_lbl = CTkLabel(current_course_frame, text="Year", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
year_lbl.grid(row=1, column=0, padx=10, sticky=W)

years = ["Select Year", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
year_combo = CTkComboBox(current_course_frame, values=years, variable=app.var_year)
year_combo.set("Select Year")
year_combo.grid(row=1, column=1, padx=2, pady=10, sticky=W)

# Semester
semester_lbl = CTkLabel(current_course_frame, text="Semester", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
semester_lbl.grid(row=1, column=2, padx=10, sticky=W)

semesters = ["Select Semester", "Semester-1", "Semester-2"]
semester_combo = CTkComboBox(current_course_frame, values=semesters, variable=app.var_semester)
semester_combo.set("Select Semester")
semester_combo.grid(row=1, column=3, padx=2, pady=10, sticky=W)

# Class Student Information
class_student_lbl = CTkLabel(left_frame, text="Class Student Information", font=("Arial", 15, "bold"), text_color="black")
class_student_lbl.grid(row=2, column=0, padx=10, pady=5, sticky=W)

class_student_frame = CTkFrame(left_frame, fg_color="white", border_width=2, border_color="black", width=750, height=340)
class_student_frame.grid(row=3, column=0, padx=10)
# class_student_frame.grid_propagate(False)

# Student ID
student_id_lbl = CTkLabel(class_student_frame, text="Student ID:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
student_id_lbl.grid(row=0, column=0, padx=10, pady=10, sticky=W)

student_id_entry = CTkEntry(class_student_frame, width=200, textvariable=app.var_std_id)
student_id_entry.grid(row=0, column=1, padx=10, pady=10, sticky=W)

# Student Name
student_name_lbl = CTkLabel(class_student_frame, text="Student Name:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
student_name_lbl.grid(row=0, column=2, padx=10, pady=10, sticky=W)

student_name_entry = CTkEntry(class_student_frame, width=200, textvariable=app.var_std_name)
student_name_entry.grid(row=0, column=3, padx=10, pady=10, sticky=W)

# Class Division
class_div_lbl = CTkLabel(class_student_frame, text="Class Division:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
class_div_lbl.grid(row=1, column=0, padx=10, pady=10, sticky=W)

div_combo = CTkComboBox(class_student_frame, values=["None", "A", "B"], variable=app.var_div)
div_combo.set("Select")
div_combo.grid(row=1, column=1, padx=10, pady=10, sticky=W)

# Roll No
roll_no_lbl = CTkLabel(class_student_frame, text="Roll No:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
roll_no_lbl.grid(row=1, column=2, padx=10, pady=10, sticky=W)

roll_no_entry = CTkEntry(class_student_frame, width=200, textvariable=app.var_roll)
roll_no_entry.grid(row=1, column=3, padx=10, pady=10, sticky=W)

# Gender
gender_lbl = CTkLabel(class_student_frame, text="Gender:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
gender_lbl.grid(row=2, column=0, padx=10, pady=10, sticky=W)

gender_combo = CTkComboBox(class_student_frame, values=["Male", "Female"], variable=app.var_gender)
gender_combo.set("Select")
gender_combo.grid(row=2, column=1, padx=10, pady=10, sticky=W)

# DOB
dob_lbl = CTkLabel(class_student_frame, text="DOB:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
dob_lbl.grid(row=2, column=2, padx=10, pady=10, sticky=W)

dob_entry = CTkEntry(class_student_frame, width=200, textvariable=app.var_dob)
dob_entry.grid(row=2, column=3, padx=10, pady=10, sticky=W)

# Email
email_lbl = CTkLabel(class_student_frame, text="Email:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
email_lbl.grid(row=3, column=0, padx=10, pady=10, sticky=W)

email_entry = CTkEntry(class_student_frame, width=200, textvariable=app.var_email)
email_entry.grid(row=3, column=1, padx=10, pady=10, sticky=W)

# Phone no
phone_lbl = CTkLabel(class_student_frame, text="Phone No:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
phone_lbl.grid(row=3, column=2, padx=10, pady=10, sticky=W)

phone_entry = CTkEntry(class_student_frame, width=200, textvariable=app.var_phone)
phone_entry.grid(row=3, column=3, padx=10, pady=10, sticky=W)

# Address
address_lbl = CTkLabel(class_student_frame, text="Address:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
address_lbl.grid(row=4, column=0, padx=10, pady=10, sticky=W)

address_entry = CTkEntry(class_student_frame, width=200, textvariable=app.var_address)
address_entry.grid(row=4, column=1, padx=10, pady=10, sticky=W)

# Teacher name
teacher_lbl = CTkLabel(class_student_frame, text="Teacher Name:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
teacher_lbl.grid(row=4, column=2, padx=10, pady=10, sticky=W)

teacher_entry = CTkEntry(class_student_frame, width=200, textvariable=app.var_teacher)
teacher_entry.grid(row=4, column=3, padx=10, pady=10, sticky=W)

# Radio Buttons
app.radiovar = StringVar(value="Other")

radiobtn1 = CTkRadioButton(class_student_frame, text="Has Photo Sample", value="Yes", variable=app.radiovar)
radiobtn1.grid(row=6, column=0, padx=10)

radiobtn2 = CTkRadioButton(class_student_frame, text="No Photo Sample", value="No", variable=app.radiovar)
radiobtn2.grid(row=6, column=1, padx=10)

# Buttons
save_btn = CTkButton(class_student_frame, text="Save", command=add_data, font=("Arial", 13, "bold"))
save_btn.grid(row=7, column=0, pady=5)

update_btn = CTkButton(class_student_frame, text="Update", command=update_data, font=("Arial", 13, "bold"))
update_btn.grid(row=7, column=1, pady=5)

delete_btn = CTkButton(class_student_frame, text="Delete", command=delete_data, font=("Arial", 13, "bold"))
delete_btn.grid(row=7, column=2, pady=5)

reset_btn = CTkButton(class_student_frame, text="Reset", command=reset_data, font=("Arial", 13, "bold"))
reset_btn.grid(row=7, column=3, pady=5)

take_photo_btn = CTkButton(class_student_frame, command=generate_dataset, text="Take Photo Sample", width=45, font=("Arial", 13, "bold"))
take_photo_btn.grid(row=8, column=0, padx=10, pady=5, sticky=W)

update_photo_btn = CTkButton(class_student_frame, text="Update Photo Sample", width=45, font=("Arial", 13, "bold"), command=update_photo_sample)
update_photo_btn.grid(row=8, column=1, padx=10, pady=5, sticky=W)

# Right label
right_lbl = CTkLabel(app, text="Class Student Information", font=("Arial", 20, "bold"), text_color="black")
right_lbl.place(x=1000, y=99)

# Right Frame
right_frame = CTkFrame(app, border_width=2, border_color="black", width=800, height=520, fg_color="white")
right_frame.place(x=1000, y=130)
right_frame.grid_propagate(False)


#===========table================
table_frame = Frame(right_frame, bd=2, bg="white", relief=RIDGE)
table_frame.place(x=10, y=10, width=770, height=440)

scroll_x = Scrollbar(table_frame, orient=HORIZONTAL)
scroll_y = Scrollbar(table_frame, orient=VERTICAL)

app.student_table = ttk.Treeview(table_frame, columns=("dep", "course", "year", "sem", "id", "name", "div", "roll", "gender", "dob", "email", "phone", "address", "teacher", "photo"), xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)

scroll_x.pack(side=BOTTOM, fill=X)
scroll_y.pack(side=RIGHT, fill=Y)

scroll_x.config(command=app.student_table.xview)
scroll_y.config(command=app.student_table.yview)

app.student_table.heading("dep", text="Department")
app.student_table.heading("course", text="Course")
app.student_table.heading("year", text="Year")
app.student_table.heading("sem", text="Semester")
app.student_table.heading("id", text="Student ID")
app.student_table.heading("name", text="Student Name")
app.student_table.heading("div", text="Class Division")
app.student_table.heading("roll", text="Roll No")
app.student_table.heading("gender", text="Gender")
app.student_table.heading("dob", text="DOB")
app.student_table.heading("email", text="Email")
app.student_table.heading("phone", text="Phone No")
app.student_table.heading("address", text="Address")
app.student_table.heading("teacher", text="Teacher Name")
app.student_table.heading("photo", text="Photo")
app.student_table["show"] = "headings"

app.student_table.column("dep", width=100)
app.student_table.column("course", width=150)
app.student_table.column("year", width=100)
app.student_table.column("sem", width=100)
app.student_table.column("id", width=100)
app.student_table.column("name", width=150)
app.student_table.column("div", width=100)
app.student_table.column("roll", width=50)
app.student_table.column("gender", width=100)
app.student_table.column("dob", width=100)
app.student_table.column("email", width=150)
app.student_table.column("phone", width=100)
app.student_table.column("address", width=100)
app.student_table.column("teacher", width=100)
app.student_table.column("photo", width=100)

app.student_table.pack(fill=BOTH, expand=1)
app.student_table.bind("<ButtonRelease-1>", get_cursor)
fetch_data()

app.mainloop()
