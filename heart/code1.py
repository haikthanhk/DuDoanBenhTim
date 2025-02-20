
from tkinter import *
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
from turtle import bgcolor
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
#import PySimpleGUI as sg


df = pd.read_csv(r'C:\Users\DELL\Documents\project\DuDoanBenhTim\heart\heart_ngoailai_roirac.csv')

data = np.array(df[['age_group_num','sex','cp','trestbps_group_num','chol_group_num','fbs','restecg','thalach_group_num','exang','oldpeak_group_num','slope','ca','thal','target']].values)
dt_Train, dt_Test = train_test_split(data, test_size=0.3 , shuffle = True)

X_train_main = dt_Train[:, :13]
y_train_main = dt_Train[:, 13]
X_test_main = dt_Test[:, :13]
y_test_main = dt_Test[:, 13]

# PCA
X = np.array(df[['age_group_num','sex','cp','trestbps_group_num','chol_group_num','fbs','restecg','thalach_group_num','exang','oldpeak_group_num','slope','ca','thal']].values)    
y = np.array(df['target'])

# PCA chuẩn bị dữ liệu
pca_features = ['age_group_num', 'chol_group_num', 'thalach_group_num', 'oldpeak_group_num', 'trestbps_group_num', 
                'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
X = np.array(df[pca_features].values)
y = np.array(df['target'])

# Định nghĩa hàm PCA_method
def PCA_method(formula):
    max_accuracy = 0
    pca_best = None
    modeImax = None
    num_pca = 0

    for j in range(1, len(pca_features) + 1):
        print(f"Lần thử: {j}")
        pca = PCA(n_components=j)
        pca.fit(X)
        Xbar = pca.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(Xbar, y, test_size=0.3, shuffle=True)

        if formula == 'id3':
            id3 = DecisionTreeClassifier(criterion='entropy', random_state=0)
            id3.fit(X_train, y_train)
            y_predict_id3 = id3.predict(X_test)
            rate = accuracy_score(y_test, y_predict_id3)
            print(f"Tỷ lệ dự đoán đúng ID3: {rate:.2f}")

            if rate > max_accuracy:
                num_pca = j
                pca_best = pca
                max_accuracy = rate
                modeImax = id3

    return modeImax, pca_best, num_pca, max_accuracy

#CHUA DUNG PCA
#id3
id3 = DecisionTreeClassifier(criterion='entropy')
id3.fit(X_train_main, y_train_main)

# Dung PCA:
#ID3
id3_PCA, pca_best_id3, num_pca_id3, max_accuracy = PCA_method('id3')


# FORM
form = tk.Tk()
form.title("Dự đoán khả năng bị bệnh tim của bệnh nhân:")
form.geometry("1700x900")



lable_people = LabelFrame(form, text = "Nhập thông tin bệnh nhân", font=("Arial Bold", 13), fg="red")
lable_people.pack(fill="both", expand="yes")
lable_people.config(bg="#FEF2D1")
# THÔNG TIN CỘT 1
lable_age = Label(form,font=("Arial Bold", 10), text = "Tuổi:" ,bg="#FEF2D1").place(x = 180 , y = 50)
lable_age_tuoi = ['thanh niên', 'trung niên', 'già', 'rất già']
lable_age = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_age_tuoi, state = "readonly")
lable_age.place(x = 410 , y = 50)
lable_age.current(0)

lable_sex = Label(form,font=("Arial Bold", 10), text = "Giới tính:" ,bg="#FEF2D1").place(x = 180 , y = 90)
lable_sex_gioitinh = ['Nam',  'Nữ']
lable_sex = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_sex_gioitinh, state = "readonly")
lable_sex.place(x = 410 , y = 90)
lable_sex.current(0)

lable_cp = Label(form,font=("Arial Bold", 10), text = "Loại đau ngực:",bg="#FEF2D1").place(x = 180 , y = 130)
lable_cp_loaidaunguc = ['Không có triệu chứng',  'Đau thắt ngực không điển hình', 'Không đau thắt ngực', 'Đau thắt ngực điển hình']  
lable_cp = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_cp_loaidaunguc, state = "readonly")
lable_cp.place(x = 410 , y = 130)
lable_cp.current(0)

lable_trestbps = Label(form,font=("Arial Bold", 10), text = "Huyết áp khi nghỉ ngơi(mm/Hg):",bg="#FEF2D1").place(x = 180 , y = 170)
lable_trestbps_huyetapkhinghi = ['thấp', 'bình thường', 'cao', 'rất cao']  
lable_trestbps = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_trestbps_huyetapkhinghi, state = "readonly")
lable_trestbps.place(x = 410 , y = 170)
lable_trestbps.current(0)

lable_chol = Label(form,font=("Arial Bold", 10), text = "Cholesterol(mg/dl):",bg="#FEF2D1").place(x = 180 , y = 210)
lable_chol_Cholesterol = ['bình thường', 'ranh giới cao', 'cao', 'rất cao']  
lable_chol = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_chol_Cholesterol, state = "readonly")
lable_chol.place(x = 410 , y = 210)
lable_chol.current(0)

lable_fbs = Label(form,font=("Arial Bold", 10), text = "Lượng đường trong máu: ",bg="#FEF2D1").place(x = 180 , y = 250)
lable_fbs_luongduong = ['<120 mg/dl',  '>120 mg/dl']
lable_fbs = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_fbs_luongduong, state = "readonly")
lable_fbs.place(x = 410 , y = 250)
lable_fbs.current(0)

lable_restecg = Label(form,font=("Arial Bold", 10), text = "Điện tâm đồ khi nghỉ ngơi:",bg="#FEF2D1").place(x = 180 , y = 290)
lable_restecg_dientamdo = ['Bình thường',  'Có sóng ST-T bất thường', 'Phì đại thất trái']
lable_restecg = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_restecg_dientamdo, state = "readonly")
lable_restecg.place(x = 410 , y = 290)
lable_restecg.current(0)

# THÔNG TIN CỘT 2
lable_thalach = Label(form,font=("Arial Bold", 10), text = "Số nhịp đập mỗi phút:",bg="#FEF2D1").place(x = 830 , y = 50)
lable_thalach_nhiptimtoida = ['rất thấp', 'thấp', 'bình thường', 'cao']
lable_thalach = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_thalach_nhiptimtoida, state = "readonly")
lable_thalach.place(x = 1080 , y = 50)
lable_thalach.current(0)

lable_exang = Label(form,font=("Arial Bold", 10), text = "Tập thể dục gây ra đau thắt ngực:",bg="#FEF2D1").place(x = 830 , y = 90)
lable_exang_daunguc = ['Không',  'Có']
lable_exang = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_exang_daunguc, state = "readonly")
lable_exang.place(x = 1080 , y = 90)
lable_exang.current(0)

lable_oldpeak = Label(form,font=("Arial Bold", 10), text = "ST trầm cảm:",bg="#FEF2D1").place(x = 830 , y = 130)
lable_oldpeak_dogiamST = ['rất thấp', 'thấp', 'bình thường', 'cao']
lable_oldpeak = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_oldpeak_dogiamST, state = "readonly")
lable_oldpeak.place(x = 1080 , y = 130)
lable_oldpeak.current(0)


lable_slope = Label(form,font=("Arial Bold", 10), text = "Độ dốc của đoạn ST:",bg="#FEF2D1").place(x = 830 , y = 170)
lable_slope_dodocST = ['Đi xuống',  'Đi lên', 'Cân bằng']
lable_slope = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_slope_dodocST, state = "readonly")
lable_slope.place(x = 1080 , y = 170)
lable_slope.current(0)

lable_ca = Label(form,font=("Arial Bold", 10), text = "Số mạch chính:",bg="#FEF2D1").place(x = 830 , y = 210)
lable_ca_somachchinh = ['0',  '1', '2', '3']
lable_ca = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_ca_somachchinh, state = "readonly")
lable_ca.place(x = 1080 , y = 210)
lable_ca.current(0)

lable_thal = Label(form,font=("Arial Bold", 10), text = "Thalassemia:",bg="#FEF2D1").place(x = 830 , y = 250)
lable_thal_Thalassemia = ['Không',  'Khuyết tật cố định', 'Lưu lượng máu bình thường', 'Khuyết tật có thể đảo ngược']
lable_thal = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_thal_Thalassemia, state = "readonly")
lable_thal.place(x = 1080 , y = 250)
lable_thal.current(0)



# KẾT QUẢ DỰ ĐOÁN
lable_people = LabelFrame(form, text = "Kết quả dự đoán", font=("Arial Bold", 13), fg="blue")
lable_people.pack(fill="both", expand="yes")
lable_people.config(bg="#FEF2D1")
# bg="#FEF2D1"

#Khi chua su dung PCA
lable_note = Label(form, text = "Khi chưa sử dụng PCA",font=("Arial Bold", 13),fg="blue",bg="#FEF2D1").place(x = 410 , y = 500)

#ID3
#dudoanid3test
y_id3 = id3.predict(X_test_main)
lbl3 = Label(form,font=("Arial Bold", 10),bg="#FEF2D1")
lbl3.place(x = 350 , y = 550)
lbl3.configure(text="Tỷ lệ dự đoán đúng của ID3: "+str(accuracy_score(y_test_main, y_id3)*100)+"%"+'\n'
                           +"Precision: "+str(precision_score(y_test_main, y_id3)*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test_main, y_id3)*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test_main, y_id3)*100)+"%"+'\n')

#khi dung PCA
lable_note = Label(form, text = "Khi sử dụng PCA",font=("Arial Bold", 13),fg="blue",bg="#FEF2D1").place(x = 930 , y = 500)

#ID3
#dudoanid3test
X_test_PCA_id3 = pca_best_id3.transform(X_test_main)
y_id3_PCA = id3_PCA.predict(X_test_PCA_id3)
lbl3 = Label(form,font=("Arial Bold", 10),bg="#FEF2D1")
lbl3.place(x = 850 , y = 550)
lbl3.configure(text="Tỷ lệ dự đoán đúng của ID3: "+str(accuracy_score(y_test_main, y_id3_PCA)*100)+"%"+'\n'
                           +"Precision: "+str(precision_score(y_test_main, y_id3_PCA)*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test_main, y_id3_PCA)*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test_main, y_id3_PCA)*100)+"%"+'\n'
                           +"Sử dụng: "+str(num_pca_id3)+"/13 trường dữ liệu")


def getValue():
    age_group_num = lable_age.get()
    if(age_group_num == 'thanh niên'):
        age_group_num = 0
    elif(age_group_num == 'trung niên'):
        age_group_num = 1
    elif(age_group_num == 'già'):
        age_group_num = 2
    elif(age_group_num == 'rất già'):
        age_group_num = 3
    sex = lable_sex.get()
    if(sex == 'Nam'):
        sex = 1
    else:
        sex = 0
    cp = lable_cp.get()
    if(cp == 'Không có triệu chứng'):
        cp = 0
    elif(cp == 'Đau thắt ngực không điển hình'):
        cp = 1
    elif(cp == 'Không đau thắt ngực'):
        cp = 2
    elif(cp == 'Đau thắt ngực điển hình'):
        cp = 3
        
    trestbps_group_num = lable_trestbps.get()
    if(trestbps_group_num == 'thấp'):
        trestbps_group_num = 0
    elif(trestbps_group_num == 'bình thường'):
        trestbps_group_num = 1
    elif(trestbps_group_num == 'cao'):
        trestbps_group_num = 2
    elif(trestbps_group_num == 'rất cao'):
        trestbps_group_num = 3
        
    chol_group_num = lable_chol.get()
    if(chol_group_num == 'bình thường'):
        chol_group_num = 0
    elif(chol_group_num == 'ranh giới cao'):
        chol_group_num = 1
    elif(chol_group_num == 'cao'):
        chol_group_num = 2
    elif(chol_group_num == 'rất cao'):
        chol_group_num = 3
        
    fbs = lable_fbs.get()
    if(fbs == '<120 mg/dl'):
        fbs = 0
    elif(fbs == '>120 mg/dl'):
        fbs = 1
        
    restecg = lable_restecg.get()
    if(restecg == 'Bình thường'):
        restecg = 0
    elif(restecg == 'Có sóng ST-T bất thường'):
        restecg = 1
    elif(restecg == 'Phì đại thất trái'):
        restecg = 2
        
    thalach_group_num = lable_thalach.get()
    if(thalach_group_num == 'rất thấp'):
        thalach_group_num = 0
    elif(thalach_group_num == 'thấp'):
        thalach_group_num = 1
    elif(thalach_group_num == 'bình thường'):
        thalach_group_num = 2
    elif(thalach_group_num == 'cao'):
        thalach_group_num = 3
        
    exang = lable_exang.get()
    if(exang == 'Có'):
        exang = 1
    else:
        exang = 0    
        
    oldpeak_group_num = lable_oldpeak.get()
    if(oldpeak_group_num == 'rất thấp'):
        oldpeak_group_num = 0
    elif(oldpeak_group_num == 'thấp'):
        oldpeak_group_num = 1
    elif(oldpeak_group_num == 'bình thường'):
        oldpeak_group_num = 2
    elif(oldpeak_group_num == 'cao'):
        oldpeak_group_num = 3
        
    slope = lable_slope.get()
    if(slope == 'Đi xuống'):
        slope = 0
    elif(slope == 'Đi lên'):
        slope = 1
    elif(slope == 'Cân bằng'):
        slope = 2
    ca = lable_ca.get()
    
    thal = lable_thal.get()
    if(thal == 'Không'):
        thal = 0
    elif(thal == 'Khuyết tật cố định'):
        thal = 1
    elif(thal == 'Lưu lượng máu bình thường'):
        thal = 2
    elif(thal == 'Khuyết tật có thể đảo ngược'):
        thal = 3
    X_dudoan = np.array([age_group_num, sex, cp, trestbps_group_num, chol_group_num, fbs,restecg, thalach_group_num, exang, oldpeak_group_num, slope, ca, thal]).reshape(1, -1)
    return X_dudoan

# dataset
def dudoanID3():
    age_group_num = lable_age.get()
    sex = lable_sex.get()
    cp = lable_cp.get()
    trestbps_group_num = lable_trestbps.get()
    chol_group_num = lable_chol.get()
    fbs = lable_fbs.get()
    restecg = lable_restecg.get()
    thalach_group_num = lable_thalach.get()
    exang = lable_exang.get()
    oldpeak_group_num = lable_oldpeak.get()
    slope = lable_slope.get()
    ca = lable_ca.get()
    thal = lable_thal.get()

    if((age_group_num == '') or (sex == '') or (cp == '') or (trestbps_group_num == '') or (chol_group_num == '') or (fbs == '') or (restecg == '') or (thalach_group_num == '') or (exang == '') or (oldpeak_group_num == '') or (slope == '') or (ca == '') or (thal == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = getValue()
        y_kqua = id3_PCA.predict(X_dudoan)
        if(y_kqua == 1):
            lbl1.configure(text= 'Yes - Bị bệnh')
        else:
            lbl1.configure(text= 'No - Không bị bệnh')

# Button
button_id3 = Button(form ,font=("Arial Bold", 10), text = 'Kết quả dự đoán theo ID3', command = dudoanID3, background="#04AA6D", foreground="white")
button_id3.place(x = 410 , y = 670)
lbl1 = Label(form, text="",font=("Arial Bold", 10),fg="blue",bg="#FEF2D1")
lbl1.place(x = 410 , y = 700)


# dataset
def dudoanID3():
    age_group_num = lable_age.get()
    sex = lable_sex.get()
    cp = lable_cp.get()
    trestbps_group_num = lable_trestbps.get()
    chol_group_num = lable_chol.get()
    fbs = lable_fbs.get()
    restecg = lable_restecg.get()
    thalach_group_num = lable_thalach.get()
    exang = lable_exang.get()
    oldpeak_group_num = lable_oldpeak.get()
    slope = lable_slope.get()
    ca = lable_ca.get()
    thal = lable_thal.get()

    if((age_group_num == '') or (sex == '') or (cp == '') or (trestbps_group_num == '') or (chol_group_num == '') or (fbs == '') or (restecg == '') or (thalach_group_num == '') or (exang == '') or (oldpeak_group_num == '') or (slope == '') or (ca == '') or (thal == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    #else:
        #if( int(age_group_num) < 1 or int(age_group_num) > 150):
            #messagebox.showerror("Thông báo", "Thông tin tuổi phải từ 0-120")
        #elif( int(trestbps_group_num) < 0 ) :
          #  messagebox.showerror("Thông báo", "Thông tin huyết áp khi nghỉ ngơi phải lớn hơn 0")
        #elif( int(chol_group_num) < 0 ) :
        ##    messagebox.showerror("Thông báo", "Thông tin cholesteron phải lớn hơn 0")
        #elif( int(thalach_group_num) < 0 ) :
         #   messagebox.showerror("Thông báo", "Thông tin nhịp tim mỗi phút phải lớn hơn 0")
        #elif( float(oldpeak_group_num) < 0 ) :
         #   messagebox.showerror("Thông báo", "Thông tin ST trầm cảm phải lớn hơn 0")
    else:
        X_dudoan = getValue()
        y_kqua = id3.predict(X_dudoan)
        if(y_kqua == 1):
            lbl1.configure(text= 'Yes - Bị bệnh')
        else:
            lbl1.configure(text= 'No - Không bị bệnh')

# Button
button_id3 = Button(form ,font=("Arial Bold", 10), text = 'Kết quả dự đoán theo ID3', command = dudoanID3, background="#04AA6D", foreground="white")
button_id3.place(x = 410 , y = 670)
lbl1 = Label(form, text="",font=("Arial Bold", 10),fg="blue", bg="#FEF2D1")
lbl1.place(x = 410 , y = 700)



def dudoanID3_PCA():
    age_group_num = lable_age.get()
    sex = lable_sex.get()
    cp = lable_cp.get()
    trestbp_group_num = lable_trestbps.get()
    chol_group_num = lable_chol.get()
    fbs = lable_fbs.get()
    restecg = lable_restecg.get()
    thalach_group_num = lable_thalach.get()
    exang = lable_exang.get()
    oldpeak_group_num = lable_oldpeak.get()
    slope = lable_slope.get()
    ca = lable_ca.get()
    thal = lable_thal.get()

    if((age_group_num == '') or (sex == '') or (cp == '') or (trestbp_group_num == '') or (chol_group_num == '') or (fbs == '') or (restecg == '') or (thalach_group_num == '') or (exang == '') or (oldpeak_group_num == '') or (slope == '') or (ca == '') or (thal == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = getValue()
        X_new = pca_best_id3.transform(X_dudoan)
        y_kqua = id3_PCA.predict(X_new)
        if(y_kqua == 1):
            lbl1_id3pca.configure(text= 'Yes - Bị bệnh')
        else:
            lbl1_id3pca.configure(text= 'No - Không bị bệnh')

# Button
button_id3_pca = Button(form ,font=("Arial Bold", 10), text = 'Kết quả dự đoán theo ID3_PCA', command = dudoanID3_PCA, background="#04AA6D", foreground="white")
button_id3_pca.place(x = 930 , y = 670)
lbl1_id3pca = Label(form, text="",font=("Arial Bold", 10),fg="blue",bg="#FEF2D1")
lbl1_id3pca.place(x = 930 , y = 700)

form.mainloop()
