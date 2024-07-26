from tkinter import *
from customtkinter import *
from tkinter import ttk
from tkinter import messagebox
import mysql.connector
import cv2
import os
import csv
from tkinter import filedialog

# CustomTkinter appearance settings
set_appearance_mode("light")
set_default_color_theme("blue")

# Initialize main application window
app = CTk()
app.title("Attendance Management System")
app.geometry("1900x700+0+0")

# Variables
app.var_atten_id = StringVar()
app.var_atten_roll = StringVar()
app.var_atten_name = StringVar()
app.var_atten_dep = StringVar()
app.var_atten_time = StringVar()
app.var_atten_date = StringVar()
app.var_atten_attendance = StringVar()

mydata = []

# Fetch data into the table
def fetchData(rows):
    app.AttendanceReportTable.delete(*app.AttendanceReportTable.get_children())
    for i in rows:
        app.AttendanceReportTable.insert("", END, values=i)

# Import CSV data
def importCsv():
    global mydata
    mydata.clear()
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Open CSV", filetypes=(("CSV File", "*.csv"), ("ALL File", "*.*")), parent=app)
    with open(fln) as myfile:
        csvread = csv.reader(myfile, delimiter=",")
        for i in csvread:
            mydata.append(i)
        fetchData(mydata)

# Export CSV data
def exportCsv():
    try:
        if len(mydata) < 1:
            messagebox.showerror("No Data", "No Data Found to Export", parent=app)
            return False
        fln = filedialog.asksaveasfilename(initialdir=os.getcwd(), title="Save CSV", filetypes=(("CSV File", "*.csv"), ("ALL File", "*.*")), parent=app)
        with open(fln, mode="w", newline="") as myfile:
            exp_write = csv.writer(myfile, delimiter=",")
            for i in mydata:
                exp_write.writerow(i)
            messagebox.showinfo("Data Export", "Your Data Exported to " + os.path.basename(fln) + " successfully", parent=app)
    except Exception as es:
        messagebox.showerror("Error", f"Due To : {str(es)}", parent=app)


# Get cursor function
def get_cursor(event=""):
    cursor_row = app.AttendanceReportTable.focus()
    content = app.AttendanceReportTable.item(cursor_row)
    row = content["values"]

    if len(row) >= 1:
        app.var_atten_id.set(row[0])
    if len(row) >= 2:
        app.var_atten_roll.set(row[1])
    if len(row) >= 3:
        app.var_atten_name.set(row[2])
    if len(row) >= 4:
        app.var_atten_dep.set(row[3])
    if len(row) >= 5:
        app.var_atten_time.set(row[4])
    if len(row) >= 6:
        app.var_atten_date.set(row[5])
    if len(row) >= 7:
        app.var_atten_attendance.set(row[6])

# Reset function
def reset_data():
    app.var_atten_id.set("")
    app.var_atten_roll.set("")
    app.var_atten_name.set("")
    app.var_atten_dep.set("")
    app.var_atten_time.set("")
    app.var_atten_date.set("")
    app.var_atten_attendance.set("Status")

# Left label
left_lbl = CTkLabel(app, text="Student Details", font=("Arial", 20, "bold"), text_color="black")
left_lbl.place(x=100, y=99)

# Left Frame
left_frame = CTkFrame(app, border_width=2, border_color="black", width=800, height=520, fg_color="white")
left_frame.place(x=100, y=130)
left_frame.grid_propagate(False)

# Left Frame Labels
left_inside_frame = CTkFrame(left_frame, fg_color="white", border_width=2, border_color="black", width=740, height=340)
left_inside_frame.grid(row=1, column=0, padx=18, pady=150, sticky=EW)

# Attendance ID
attendanceId_lbl = CTkLabel(left_inside_frame, text="Attendance ID:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
attendanceId_lbl.grid(row=0, column=0, padx=10, pady=10, sticky=W)

attendanceId_entry = CTkEntry(left_inside_frame, width=200, textvariable=app.var_atten_id)
attendanceId_entry.grid(row=0, column=1, padx=10, pady=10, sticky=W)

# Roll
roll_lbl = CTkLabel(left_inside_frame, text="Roll:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
roll_lbl.grid(row=0, column=2, padx=10, pady=10, sticky=W)

roll_entry = CTkEntry(left_inside_frame, width=200, textvariable=app.var_atten_roll)
roll_entry.grid(row=0, column=3, padx=10, pady=10, sticky=W)

# Name
name_lbl = CTkLabel(left_inside_frame, text="Name:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
name_lbl.grid(row=1, column=0, padx=10, pady=10, sticky=W)

name_entry = CTkEntry(left_inside_frame, width=200, textvariable=app.var_atten_name)
name_entry.grid(row=1, column=1, padx=10, pady=10, sticky=W)

# Department
dep_lbl = CTkLabel(left_inside_frame, text="Department:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
dep_lbl.grid(row=1, column=2, padx=10, pady=10, sticky=W)

dep_entry = CTkEntry(left_inside_frame, width=200, textvariable=app.var_atten_dep)
dep_entry.grid(row=1, column=3, padx=10, pady=10, sticky=W)

# Time
time_lbl = CTkLabel(left_inside_frame, text="Time:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
time_lbl.grid(row=2, column=0, padx=10, pady=10, sticky=W)

time_entry = CTkEntry(left_inside_frame, width=200, textvariable=app.var_atten_time)
time_entry.grid(row=2, column=1, padx=10, pady=10, sticky=W)

# Date
date_lbl = CTkLabel(left_inside_frame, text="Date:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
date_lbl.grid(row=2, column=2, padx=10, pady=10, sticky=W)

date_entry = CTkEntry(left_inside_frame, width=200, textvariable=app.var_atten_date)
date_entry.grid(row=2, column=3, padx=10, pady=10, sticky=W)

# Attendance Status
attendance_lbl = CTkLabel(left_inside_frame, text="Attendance Status:", font=("Arial", 13, "bold"), text_color="black", fg_color="white")
attendance_lbl.grid(row=3, column=0, padx=10, pady=10, sticky=W)

app.atten_status = CTkComboBox(left_inside_frame, variable=app.var_atten_attendance, values=["Status", "Present", "Absent"], state="readonly")
app.atten_status.set("Status")
app.atten_status.grid(row=3, column=1, padx=10, pady=10, sticky=W)

# Buttons
import_btn = CTkButton(left_inside_frame, text="Import CSV", command=importCsv, font=("Arial", 13, "bold"))
import_btn.grid(row=4, column=0, padx=10, pady=5)

export_btn = CTkButton(left_inside_frame, text="Export CSV", command=exportCsv, font=("Arial", 13, "bold"))
export_btn.grid(row=4, column=1, padx=10, pady=5)

reset_btn = CTkButton(left_inside_frame, text="Reset", command=reset_data, font=("Arial", 13, "bold"))
reset_btn.grid(row=4, column=2, padx=10, pady=5)

# Right label
right_lbl = CTkLabel(app, text="Attendance Details", font=("Arial", 20, "bold"), text_color="black")
right_lbl.place(x=1000, y=99)

# Right Frame
right_frame = CTkFrame(app, border_width=2, border_color="black", width=800, height=520, fg_color="white")
right_frame.place(x=1000, y=130)
right_frame.grid_propagate(False)

table_frame = Frame(right_frame, bd=2, relief=RIDGE, bg="white")
table_frame.place(x=10, y=10, width=780, height=440)

# Scroll bar table
scroll_x = ttk.Scrollbar(table_frame, orient=HORIZONTAL)
scroll_y = ttk.Scrollbar(table_frame, orient=VERTICAL)
app.AttendanceReportTable = ttk.Treeview(table_frame, columns=("id", "roll", "name", "department", "time", "date", "attendance"), xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)

scroll_x.pack(side=BOTTOM, fill=X)
scroll_y.pack(side=RIGHT, fill=Y)

scroll_x.config(command=app.AttendanceReportTable.xview)
scroll_y.config(command=app.AttendanceReportTable.yview)

app.AttendanceReportTable.heading("id", text="Attendance ID")
app.AttendanceReportTable.heading("roll", text="Roll")
app.AttendanceReportTable.heading("name", text="Name")
app.AttendanceReportTable.heading("department", text="Department")
app.AttendanceReportTable.heading("time", text="Time")
app.AttendanceReportTable.heading("date", text="Date")
app.AttendanceReportTable.heading("attendance", text="Attendance")

app.AttendanceReportTable["show"] = "headings"
app.AttendanceReportTable.column("id", width=100)
app.AttendanceReportTable.column("roll", width=100)
app.AttendanceReportTable.column("name", width=100)
app.AttendanceReportTable.column("department", width=100)
app.AttendanceReportTable.column("time", width=100)
app.AttendanceReportTable.column("date", width=100)
app.AttendanceReportTable.column("attendance", width=100)

app.AttendanceReportTable.pack(fill=BOTH, expand=1)
app.AttendanceReportTable.bind("<ButtonRelease-1>", get_cursor)

# Start the main application loop
app.mainloop()
