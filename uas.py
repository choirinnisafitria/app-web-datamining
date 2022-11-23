import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib

st.title("PENAMBANGAN DATA")
st.write("##### Nama  : Choirinnisa' Fitria ")
st.write("##### Nim   : 2004111000149 ")
st.write("##### Kelas : Penambangan Data C ")
description, upload_data, preporcessing, modeling, implementation = st.tabs(["Description", "Upload Data", "Prepocessing", "Modeling", "Implementation"])

with description:
    st.write("###### Data Set : Human Stress Detection in and through Sleep - Deteksi Stres Manusia di dalam dan melalui Tidur ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep?select=SaYoPillow.csv")
    st.write("""###### Tentang Data Set :""")
    st.write(""" Mengingat gaya hidup saat ini, orang hanya tidur melupakan manfaat tidur bagi tubuh manusia. Bantal Smart-Yoga (SaYoPillow) diusulkan untuk membantu dalam memahami hubungan antara stres dan tidur dan untuk sepenuhnya mewujudkan gagasan "Smart-Sleeping" dengan mengusulkan perangkat edge. Prosesor tepi dengan model yang menganalisis perubahan fisiologis yang terjadi selama tidur bersama dengan kebiasaan tidur diusulkan. Berdasarkan perubahan ini selama tidur, prediksi stres untuk hari berikutnya diusulkan. Transfer aman dari data stres yang dianalisis bersama dengan perubahan fisiologis rata-rata ke cloud IoT untuk penyimpanan diimplementasikan. Transfer aman data apa pun dari cloud ke aplikasi pihak ketiga mana pun juga diusulkan. Antarmuka pengguna disediakan yang memungkinkan pengguna untuk mengontrol aksesibilitas dan visibilitas data.
    Di SayoPillow.csv, Anda akan melihat hubungan antara parameter - kisaran mendengkur pengguna, laju pernapasan, suhu tubuh, laju pergerakan tungkai, kadar oksigen darah, pergerakan mata, jumlah jam tidur, detak jantung, dan Tingkat Stres (0 - rendah/normal, 1 – sedang rendah, 2-sedang, 3-sedang tinggi, 4-tinggi) yang dihasilkan dari Tinjauan Pustaka. (Tidak ada subjek manusia yang dipertimbangkan)
    Jika Anda menggunakan kumpulan data ini atau menemukan informasi ini berkontribusi terhadap penelitian Anda, silakan kutip:
    1. L. Rachakonda, AK Bapatla, SP Mohanty, dan E. Kougianos, “SaYoPillow: Kerangka Kerja IoMT Terintegrasi-Privasi-Terintegrasi Blockchain untuk Manajemen Stres Mempertimbangkan Kebiasaan Tidur”, Transaksi IEEE pada Elektronik Konsumen (TCE), Vol. 67, No. 1, Feb 2021, hlm. 20-29.
    2. L. Rachakonda, SP Mohanty, E. Kougianos, K. Karunakaran, dan M. Ganapathiraju, “Bantal Cerdas: Perangkat Berbasis IoT untuk Deteksi Stres Mempertimbangkan Kebiasaan Tidur”, dalam Prosiding Simposium Internasional IEEE ke-4 tentang Sistem Elektronik Cerdas ( iSES), 2018, hlm. 161--166.""")
    st.write("###### Aplikasi ini untuk : Deteksi Stres Manusia di dalam dan melalui Tidur ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/choirinnisafitria/Web-Data-Mining ")

with upload_data:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with preporcessing:
    st.write("""# Preprocessing""")
    df[["sr", "rr", "t", "lm", "bo", "rem", "sr.1", "hr"]].agg(['min','max'])

    df.sl.value_counts()
    # df = df.drop(columns=["date"])

    X = df.drop(columns="sl")
    y = df.sl
    "### Membuang fitur yang tidak diperlukan"
    df

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    "### Transformasi Label"
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.sl).columns.values.tolist()

    "### Label"
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, y.shape

    le.inverse_transform(y)

    labels = pd.get_dummies(df.sl).columns.values.tolist()
    
    "### Label"
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X

    X.shape, y.shape

with modeling:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)

with implementation:
    st.write("# Implementation")
    Snoring_Rate = st.number_input('Masukkan tingkat mendengkur : ')
    Respiration_Rate = st.number_input('Masukkan laju respirasi : ')
    Body_Temperature = st.number_input('Masukkan suhu tubuh : ')
    Limb_Movement = st.number_input('Masukkan gerakan ekstremitas : ')
    Blood_Oxygen = st.number_input('Masukkan oksigen darah : ')
    Eye_Movement = st.number_input('Masukkan gerakan mata : ')
    Sleeping_Hours = st.number_input('Masukkan jam tidur : ')
    Heart_Rate = st.number_input('Masukkan detak jantung : ')

    def submit():
        # input
        inputs = np.array([[
            Snoring_Rate,
            Respiration_Rate,
            Body_Temperature,
            Limb_Movement,
            Blood_Oxygen,
            Eye_Movement,
            Sleeping_Hours,
            Heart_Rate
            ]])
        le = joblib.load("le.save")
        model1 = joblib.load("knn.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang di masukkan, maka Deteksi Stres Manusia di dalam dan melalui Tidur : {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()

