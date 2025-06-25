import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import joblib


warnings.filterwarnings('ignore')

st.set_page_config(page_title="Segmentasi Kemiskinan", layout="wide")

# Fungsi load data awal
@st.cache_data
def load_data():
    df = pd.read_excel("data_bersih.xlsx")
    return df

# Halaman Database
def page_database():
    st.title("📁 Database Segmentasi Kemiskinan")
    df = load_data()

    if not df.empty:
        st.subheader("🔍 Pencarian Berdasarkan Nama")
        kolom_nama = st.selectbox("Pilih kolom untuk pencarian:", df.columns)
        input_nama = st.text_input("Masukkan kata kunci pencarian (tidak case-sensitive):")

        if input_nama:
            hasil = df[df[kolom_nama].astype(str).str.contains(input_nama, case=False, na=False)]
            st.subheader("📌 Hasil Pencarian")
            if not hasil.empty:
                st.success(f"{len(hasil)} data ditemukan.")
                st.dataframe(hasil)
            else:
                st.warning("Data tidak ditemukan.")
        else:
            st.subheader("📋 Pratinjau Seluruh Data")
            st.dataframe(df)
    else:
        st.warning("Data kosong atau gagal dimuat.")

# Halaman Proses Segmentasi
def page_segmentasi():
    st.title("⚙️ Proses Segmentasi dengan Bobot AHP, SAW, dan KMeans")

    uploaded_file = st.file_uploader("Upload file Excel yang akan digunakan", type=["xlsx"])
    if uploaded_file is None:
        st.warning("Silakan upload file Excel terlebih dahulu.")
        return
    
    # Baca data
    df = pd.read_excel(uploaded_file)

    # Pisahkan kolom numerik dan non-numerik
    fitur_numerik = df.select_dtypes(include=['number']).columns.tolist()
    fitur_non_numerik = df.select_dtypes(exclude=['number']).columns.tolist()

    # if len(fitur_numerik) < 14:
    #     st.error(f"Data harus punya minimal 14 kolom numerik, sekarang hanya {len(fitur_numerik)} kolom.")
    #     return

    st.subheader("📋 Data yang Diupload")
    st.dataframe(df)

    # Bobot AHP tetap
    ahp_weights = [
        0.1924, 0.1478, 0.1339, 0.0979, 0.0773, 0.0737, 0.0630, 0.0488,
        0.0414, 0.0353, 0.0282, 0.0236, 0.0195, 0.0173
    ]

    # Ambil hanya 14 kolom numerik pertama untuk SAW
    fitur_terpilih = fitur_numerik[:14]
    df_saw = df.copy()

    # Normalisasi sebagai cost: min / x_ij
    X = df_saw[fitur_terpilih].astype(float)
    X_norm = X.copy()
    
    for col in X.columns:
        X_norm[col] = X[col].min() / X[col]
    
    # Kalikan dengan bobot AHP
    for i, col in enumerate(fitur_terpilih):
        X_norm[col] *= ahp_weights[i]
    
    # Hitung skor SAW
    df_saw['SAW_Score'] = X_norm.sum(axis=1)

    st.subheader("✅ Data dengan Nilai SAW")
    st.dataframe(df_saw)

    # Slider pemilihan cluster
    n_clusters = st.slider("Pilih jumlah cluster (KMeans)", min_value=2, max_value=10, value=3)

    # KMeans clustering berdasarkan SAW_Score
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_saw['Cluster'] = kmeans.fit_predict(df_saw[['SAW_Score']]) + 1

    st.subheader("🔍 Hasil Segmentasi (SAW + KMeans)")
    st.dataframe(df_saw)

    # Simpan hasil untuk halaman dashboard
    st.session_state['data_proses'] = df_saw

    # Opsi download
    if st.button("📥 Download hasil segmentasi"):
        output_file = "hasil_segmentasi.xlsx"
        df_saw.to_excel(output_file, index=False)
        st.success(f"File berhasil disimpan: {output_file}")


def page_dashboard():
    st.title("📊 Dashboard Segmentasi Kemiskinan")

    if 'data_proses' not in st.session_state:
        st.warning("Data hasil segmentasi belum tersedia. Silakan lakukan proses segmentasi terlebih dahulu.")
        return

    # Ambil data awal dan data hasil segmentasi
    df_awal = load_data()
    df_proses = st.session_state['data_proses']

    # Ambil 2 kolom terakhir dari data proses (SAW_Score dan Cluster)
    df_saw_summary = df_proses.iloc[:, -2:]

    df_saw_summary = df_saw_summary.reset_index(drop=True)
    df_awal = df_awal.reset_index(drop=True)

    # Gabungkan data awal dengan hasil segmentasi
    df = pd.concat([df_awal, df_saw_summary], axis=1)

    if df.empty:
        st.warning("Data kosong atau gagal dimuat.")
        return

    # Checkbox visualisasi berdasarkan cluster
    use_cluster = False
    if 'Cluster' in df.columns:
        use_cluster = st.checkbox("Tampilkan Visualisasi Berdasarkan Cluster", value=True)

    # ========================
    # Bar Chart Kolom Kategorikal / Ordinal
    st.subheader("📈 Frekuensi Data Berdasarkan Kolom")

    possible_cat_cols = [col for col in df.columns if df[col].dtype in ['object', 'category', 'float64', 'int64']]
    
    bar_col = st.selectbox("Pilih Kolom:", options=possible_cat_cols)
    
    if bar_col:
        if use_cluster and bar_col != 'Cluster':
            freq_df = df.groupby(['Cluster', bar_col]).size().reset_index(name='Jumlah')
            fig = px.bar(freq_df, x=bar_col, y='Jumlah', color='Cluster', barmode='group',
                         title=f"Distribusi {bar_col} berdasarkan Cluster")
        else:
            freq_df = df[bar_col].value_counts().reset_index()
            freq_df.columns = [bar_col, 'Jumlah']
            fig = px.bar(freq_df, x=bar_col, y='Jumlah',
                         title=f"Distribusi {bar_col}")
    
        st.plotly_chart(fig, use_container_width=True)
    
        with st.expander("📄 Lihat Data Tabel"):
            st.dataframe(freq_df)
            csv = freq_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", data=csv, file_name="bar_chart_data.csv", mime="text/csv")


    # Distribusi Pendapatan
    if 'Pendapatan' in df.columns:
        st.subheader("💰 Analisis Distribusi Pendapatan")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Histogram", "📦 Box Plot", "🎻 Violin Plot", "🌊 KDE Plot"
        ])

        with tab1:
            fig = px.histogram(df, x='Pendapatan', nbins=30,
                               color='Cluster' if use_cluster else None,
                               title="Histogram Pendapatan",
                               labels={'Pendapatan': 'Pendapatan (Rupiah)'})
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = px.box(df, x='Cluster' if use_cluster else None, y='Pendapatan',
                         color='Cluster' if use_cluster else None,
                         title="Box Plot Pendapatan")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = px.violin(df, x='Cluster' if use_cluster else None, y='Pendapatan',
                            color='Cluster' if use_cluster else None, box=True, points='all',
                            title="Violin Plot Pendapatan")
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            import seaborn as sns
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            if use_cluster:
                for cluster in sorted(df['Cluster'].unique()):
                    subset = df[df['Cluster'] == cluster]
                    sns.kdeplot(subset['Pendapatan'], label=f'Cluster {cluster}', ax=ax, shade=True)
                ax.legend()
            else:
                sns.kdeplot(df['Pendapatan'], shade=True, ax=ax)
            ax.set_title("KDE Plot Pendapatan")
            ax.set_xlabel("Pendapatan (Rupiah)")
            st.pyplot(fig)

        
    # ========================
    # Sebaran Nilai Skor SAW per Kecamatan
    
    if 'Kecamatan' in df.columns and 'SAW_Score' in df.columns:
        st.subheader("📌 Sebaran Nilai Skor SAW di Tiap Kecamatan")

        fig_saw_box = px.box(df, x='Kecamatan', y='SAW_Score', points='all',
                             title="Sebaran Skor SAW per Kecamatan",
                             labels={'SAW_Score': 'Skor SAW'})
        fig_saw_box.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_saw_box, use_container_width=True)

        st.subheader("🏆 Kecamatan dengan Rata-rata Skor SAW Tertinggi & Terendah")

        mean_saw = df.groupby('Kecamatan')['SAW_Score'].mean().reset_index()
        mean_saw = mean_saw.sort_values('SAW_Score', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📈 Tertinggi", value=mean_saw.iloc[0]['Kecamatan'], delta=f"{mean_saw.iloc[0]['SAW_Score']:.4f}")
        with col2:
            st.metric("📉 Terendah", value=mean_saw.iloc[-1]['Kecamatan'], delta=f"{mean_saw.iloc[-1]['SAW_Score']:.4f}")
        
        fig_mean_saw = px.bar(
            mean_saw,
            x='Kecamatan',
            y='SAW_Score',
            color='SAW_Score',
            color_continuous_scale='Reds_r',
            title="Rata-rata Skor SAW per Kecamatan",
            labels={'SAW_Score': 'Rata-rata Skor SAW'}
        )
        fig_mean_saw.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_mean_saw, use_container_width=True)


    # ========================
    # Proporsi Keluarga per Cluster di Tiap Kecamatan

    if 'Cluster' in df.columns and 'Kecamatan' in df.columns:
        st.subheader("🚨 Proporsi Keluarga per Cluster di Tiap Kecamatan")
    
        cluster_counts = df.groupby(['Kecamatan', 'Cluster']).size().reset_index(name='Jumlah')
        total_counts = df.groupby('Kecamatan').size().reset_index(name='Total')
    
        # Gabungkan jumlah cluster dengan total populasi
        merged = pd.merge(cluster_counts, total_counts, on='Kecamatan')
        merged['Proporsi'] = merged['Jumlah'] / merged['Total']
    
        # Buat tab untuk setiap cluster unik
        tabs = st.tabs([f"Cluster {cl}" for cl in sorted(df['Cluster'].unique())])
    
        for i, cluster_val in enumerate(sorted(df['Cluster'].unique())):
            with tabs[i]:
                df_cluster = merged[merged['Cluster'] == cluster_val]
    
                fig_cluster = px.bar(df_cluster, x='Kecamatan', y='Proporsi',
                                     title=f"Proporsi Keluarga di Cluster {cluster_val}",
                                     labels={'Proporsi': 'Proporsi'},
                                     color='Proporsi', color_continuous_scale='blues')
                fig_cluster.update_layout(xaxis_tickangle=-45, yaxis_tickformat='.0%')
                st.plotly_chart(fig_cluster, use_container_width=True)
    
                with st.expander("📄 Lihat Tabel Data"):
                    st.dataframe(df_cluster[['Kecamatan', 'Jumlah', 'Total', 'Proporsi']])
                    csv_cluster = df_cluster.to_csv(index=False).encode('utf-8')
                    st.download_button(f"📥 Download CSV Cluster {cluster_val}", data=csv_cluster,
                                       file_name=f"proporsi_cluster_{cluster_val}.csv", mime="text/csv")


        # ========================
        # Statistik Tambahan
 
        st.subheader("📋 Statistik Tambahan SAW & Cluster")

        # 1. 5 Kecamatan dengan rata-rata skor SAW tertinggi
        top5_mean_high = mean_saw.head(5)
        st.markdown("#### 📈 5 Kecamatan dengan Rata-rata Skor SAW Tertinggi")
        st.dataframe(top5_mean_high)

        # 2. 5 Kecamatan dengan rata-rata skor SAW terendah
        top5_mean_low = mean_saw.tail(5)
        st.markdown("#### 📉 5 Kecamatan dengan Rata-rata Skor SAW Terendah")
        st.dataframe(top5_mean_low)

        # 3. 5 Kecamatan dengan skor SAW individu terendah
        lowest_saw_individual = df.sort_values('SAW_Score').head(5)
        st.markdown("#### 🚨 5 Kecamatan dengan Skor SAW Terendah (Individu)")
        st.dataframe(lowest_saw_individual[['Kecamatan', 'SAW_Score']])

        # 4. 10 Kecamatan paling terpolarisasi (Cluster 0 dan Cluster max)
        cluster_counts = df.groupby(['Kecamatan', 'Cluster']).size().unstack(fill_value=0)
    
        # Hitung proporsi cluster per kecamatan
        cluster_props = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)
        
        # Tentukan cluster minimal dan maksimal (cluster ekstrem)
        cluster_min = cluster_props.columns.min()
        cluster_max = cluster_props.columns.max()
        
        # Hitung Polarization sebagai selisih absolut proporsi cluster ekstrem per kecamatan
        cluster_props['Polarization'] = abs(cluster_props[cluster_max] - cluster_props[cluster_min])
        
        # Urutkan kecamatan berdasarkan Polarization menurun
        polarized_kecamatans = cluster_props['Polarization'].sort_values(ascending=False).reset_index()
        
        # Ambil top 10 kecamatan paling terpolarisasi
        top10_polar = polarized_kecamatans.head(10)
        
        # Tampilkan di Streamlit
        st.markdown("#### 🧭 Top 10 Kecamatan Paling Terpolarisasi antara Cluster Terendah dan Tertinggi")
        st.dataframe(top10_polar.rename(columns={
            'Polarization': 'Polarization (Selisih Proporsi)',
            'Kecamatan': 'Kecamatan'
        }))

        # Fungsi interpretasi proporsi
        def interpretasi_proporsi(prop_series, var_name, mapping=None):
            if mapping:
                prop_series.index = prop_series.index.map(mapping)
        
            max_val = prop_series.idxmax()
            max_prop = prop_series.max()
        
            interpretasi = f"Mayoritas responden untuk variabel **{var_name}** berada pada kategori **{max_val}** dengan proporsi sebesar **{max_prop:.1f}%**."
        
            if len(prop_series) > 1:
                sorted_props = prop_series.sort_values(ascending=False)
                second_val = sorted_props.index[1]
                second_prop = sorted_props.iloc[1]
                if second_prop > 20:
                    interpretasi += f" Kategori berikutnya adalah **{second_val}** dengan proporsi sebesar **{second_prop:.1f}%**."
        
            return interpretasi
        

        st.markdown("## 📊 Proporsi Variabel Kemiskinan Berdasarkan Cluster")
        indikator_cols = [
            'FrekuensiMakanPerHari', 
            'BisaBerobatKePuskesmas', 
            'Pendapatan', 
            'SumberPenerangan',
            'BahanBakarMemasak', 
            'MemilikifasilitasBuangAirBesar', 
            'FrekuensiMakanDagingSusuAyam',
            'LuasLantai', 
            'JenisDinding', 
            'SumberAirMinum', 
            'MemilikiSimpanan',
            'JenisLantai', 
            'FrekuensiBeliPakaianBaru', 
            'Pendidikan'
        ]
        
        # Label mapping
        label_mapping = {
            'Pendidikan': {
                1: 'Perguruan Tinggi', 
                2: 'SMA', 
                3: 'SMP', 
                4: 'SD', 
                5: 'Tidak Sekolah'},
            'MemilikifasilitasBuangAirBesar': {
                1: 'Ya, dengan Septic Tank',
                2: 'Ya, tanpa Septic Tank', 
                3: 'Tidak, Jamban Umum/Bersama'},
            'SumberPenerangan': {
                1: 'Listrik PLN', 
                2: 'Listrik Non-PLN'},
            'BahanBakarMemasak': {
                1: 'Listrik/Gas', 
                2: 'Minyak Tanah', 
                3: 'Kayu/Arang/Lainnya'},
            'JenisLantai': {
                1: 'Keramik/Granit/Marmer/Ubin/Tegel/Teraso', 
                2: 'Semen', 
                3: 'Kayu/Papan', 
                4: 'Bambu', 
                5: 'Tanah'},
            'JenisDinding': {
                1: 'Tembok', 
                2: 'Seng', 
                3: 'Kayu/Papan', 
                4: 'Bambu'},
            'SumberAirMinum': {
                1: 'Ledeng/PAM/Kemasan/Sumur Bor/Terlindung',
                2: 'Sumur Tidak Terlindung',
                3: 'Air Permukaan/Lainnya',
                4: 'Air Hujan'},
            'MemilikiSimpanan': {1: 'Ya', 2: 'Tidak'},
            'BisaBerobatKePuskesmas': {1: 'Ya', 2: 'Tidak'},
            'FrekuensiMakanPerHari': {
                1: '1 Kali', 
                2: '2 Kali', 
                3: '3 Kali atau Lebih'},
            'FrekuensiMakanDagingSusuAyam': {
                1: '1 Kali/Jarang', 
                2: '2 Kali/Kadang',
                3: '3 Kali atau Lebih/Sering'},
            'FrekuensiBeliPakaianBaru': {
                0: '<1x/tahun', 
                1: '1x/tahun', 
                2: '2x/tahun', 
                3: '≥3x/tahun'},
            'Pendapatan': {1: '>600k', 2: '<=600k'},
            'LuasLantai': {1: '>8m2', 2: '<8m2'}
        }
        
        # Buat tab untuk tiap cluster
        tabs = st.tabs([f"Cluster {i}" for i in sorted(df_proses['Cluster'].unique())])
        
        # Loop setiap cluster
        for i, cluster_id in enumerate(sorted(df_proses['Cluster'].unique())):
            with tabs[i]:
                st.markdown(f"### 🔍 Distribusi Variabel di Cluster {cluster_id}")
                df_cluster = df_proses[df_proses['Cluster'] == cluster_id]
        
                for col in indikator_cols:
                    st.markdown(f"**{col}**")
        
                    # Hitung proporsi
                    prop = df_cluster[col].value_counts(normalize=True).sort_index() * 100
        
                    # Mapping label (jika tersedia)
                    mapped_prop = prop.copy()
                    if col in label_mapping:
                        mapped_prop.index = mapped_prop.index.map(label_mapping[col])
        
                    # Buat dataframe proporsi
                    df_prop = mapped_prop.reset_index()
                    df_prop.columns = [col, 'Proporsi (%)']
                    df_prop['Proporsi (%)'] = df_prop['Proporsi (%)'].round(2)
        
                    # Tampilkan tabel
                    st.dataframe(df_prop, use_container_width=True)
        
                    # Interpretasi otomatis
                    interpretasi = interpretasi_proporsi(prop, col, label_mapping.get(col))
                    st.markdown(f"> {interpretasi}")


def page_inputasi():
    st.title("✍️ Input Data Segmentasi Kemiskinan")
    st.write("Silakan isi form berikut untuk menginput data keluarga yang akan disegmentasi.")

    # Load data historis
    try:
        data_proses = pd.read_excel("data_proses2.xlsx")
    except Exception as e:
        st.error("Gagal memuat data historis: " + str(e))
        return

    columns_input = data_proses.columns[3:17]  # Kolom numerik untuk SAW

    with st.form("input_form"):
        kepala_keluarga = st.text_input("Nama Kepala Keluarga")
        kecamatan = st.text_input("Kecamatan")
        desa = st.text_input("Desa/Kelurahan")
        frekuensi_makan = st.number_input("Frekuensi Makan per Hari", min_value=0, max_value=5, value=3)
        bisa_berobat = st.radio("Bisa Berobat ke Puskesmas?", ["Ya", "Tidak"])
        pendapatan = st.number_input("Pendapatan Bulanan (Rp)", min_value=0, step=50000)

        sumber_penerangan = st.selectbox("Sumber Penerangan", [
            'Listrik Pribadi s/d 900 Watt', 
            'Listrik Pribadi > 900 Watt',
            'Non-Listrik', 
            'Listrik Bersama', 
            'Genset/solar cell'])

        bahan_bakar = st.selectbox("Bahan Bakar Memasak", [
            'Listrik/Gas', 
            'Minyak Tanah', 
            'Arang/Kayu', 
            'Lainnya'])

        fasilitas_bab = st.selectbox("Fasilitas Buang Air Besar", [
            'Ya, dengan Septic Tank', 
            'Ya, tanpa Septic Tank',
            'Tidak, Jamban Umum/Bersama', 
            'Lainnya'])

        frek_makan_daging = st.number_input("Frekuensi Makan Daging/Susu/Ayam per Minggu", min_value=0, max_value=21, value=1)
        luas_lantai = st.number_input("Luas Lantai (m²)", min_value=0)

        jenis_dinding = st.selectbox("Jenis Dinding", [
            'Tembok', 
            'Seng', 
            'Lainnya', 
            'Kayu/Papan', 
            'Bambu'])

        sumber_air_minum = st.selectbox("Sumber Air Minum", [
            'Ledeng/PAM', 
            'Air Kemasan/Isi Ulang', 
            'Sumur Bor', 
            'Sumur Terlindung',
            'Sumur Tidak Terlindung', 
            'Air Permukaan (Sungai, Danau, dll)', 
            'Air Hujan', 'Lainnya'])

        memiliki_simpanan = st.radio("Memiliki Simpanan?", ["Ya", "Tidak"])

        jenis_lantai = st.selectbox("Jenis Lantai", [
            'Keramik/Granit/Marmer/Ubin/Tegel/Teraso', 
            'Semen', 
            'Lainnya', 
            'Kayu/Papan', 
            'Bambu', 
            'Tanah'])

        frek_beli_pakaian = st.number_input("Frekuensi Beli Pakaian Baru per Tahun", min_value=0, max_value=20, value=1)

        pendidikan = st.selectbox("Pendidikan Tertinggi", [
            'Tidak/belum sekolah', 
            'Tidak tamat SD/sederajat', 
            'Siswa SD/sederajat', 
            'Tamat SD/sederajat',
            'Siswa SMP/sederajat', 
            'Tamat SMP/sederajat', 
            'Siswa SMA/sederajat', 
            'Tamat SMA/sederajat',
            'Mahasiswa Perguruan Tinggi', 
            'Tamat Perguruan Tinggi'])

        submitted = st.form_submit_button("Proses")

    if submitted:
        try:
            # Load model KMeans
            kmeans = joblib.load("kmeans_model.pkl")

            # Encode input data
            nilai_input = [
                kepala_keluarga,
                kecamatan,
                desa,
                frekuensi_makan,
                1 if bisa_berobat == "Ya" else 2,
                2 if pendapatan <= 600_000 else 1,
                {'Listrik Pribadi s/d 900 Watt': 1, 
                 'Listrik Pribadi > 900 Watt': 1,
                 'Non-Listrik': 2, 
                 'Listrik Bersama': 2, 
                 'Genset/solar cell': 2}
                [sumber_penerangan],
                {'Listrik/Gas': 1, 
                 'Minyak Tanah': 2, 
                 'Arang/Kayu': 3, 
                 'Lainnya': 2}
                [bahan_bakar],
                {'Ya, dengan Septic Tank': 1, 
                 'Ya, tanpa Septic Tank': 2,
                 'Tidak, Jamban Umum/Bersama': 3, 
                 'Lainnya': 2}
                [fasilitas_bab],
                2 if frek_makan_daging <= 1 else 1,
                2 if luas_lantai <= 8 else 1,
                {'Tembok': 1, 
                 'Seng': 2, 
                 'Lainnya': 2, 
                 'Kayu/Papan': 3, 'Bambu': 4}
                [jenis_dinding],
                {'Ledeng/PAM': 1, 
                 'Air Kemasan/Isi Ulang': 1, 
                 'Sumur Bor': 1, 
                 'Sumur Terlindung': 1,
                 'Sumur Tidak Terlindung': 2, 
                 'Air Permukaan (Sungai, Danau, dll)': 3,
                 'Air Hujan': 4, 
                 'Lainnya': 3}
                [sumber_air_minum],
                1 if memiliki_simpanan == "Ya" else 2,
                {'Keramik/Granit/Marmer/Ubin/Tegel/Teraso': 1, 
                 'Semen': 2,
                 'Lainnya': 3, 
                 'Kayu/Papan': 3, 
                 'Bambu': 4, 
                 'Tanah': 5}
                [jenis_lantai],
                2 if frek_beli_pakaian <= 1 else 1,
                {'Tidak/belum sekolah': 5, 
                 'Tidak tamat SD/sederajat': 5,
                 'Siswa SD/sederajat': 4, 
                 'Tamat SD/sederajat': 4,
                 'Siswa SMP/sederajat': 3, 
                 'Tamat SMP/sederajat': 3,
                 'Siswa SMA/sederajat': 2, 
                 'Tamat SMA/sederajat': 2,
                 'Mahasiswa Perguruan Tinggi': 1, 
                 'Tamat Perguruan Tinggi': 1}[pendidikan]
            ]

            # Buat dataframe baru
            df_new = pd.DataFrame([nilai_input], columns=data_proses.columns[:17])
            df_all = pd.concat([data_proses.iloc[:, :17], df_new], ignore_index=True)

            # Normalisasi (Min-Max)
            df_norm = df_all.copy()
            columns_input = df_all.columns[3:17]
            print("c",columns_input)
            # Normalisasi seluruh kolom sebagai cost
            for col in columns_input:
                df_norm[col] = df_norm[col].min() / df_norm[col]

            # Hitung skor SAW
            bobot = [
                0.1924, 0.1478, 0.1339, 0.0979, 0.0773, 0.0737,
                0.0630, 0.0488, 0.0414, 0.0353, 0.0282, 0.0236,
                0.0195, 0.0173
            ]
            nilai_normalisasi = df_norm.iloc[-1, 3:17].values
            saw_score = sum([nilai_normalisasi[i] * bobot[i] for i in range(14)])
            print("a:", nilai_normalisasi)
            # Prediksi cluster
            cluster_label = int(kmeans.predict([[saw_score]])[0])

            # Format hasil akhir
            hasil_akhir = pd.DataFrame([{
                **dict(zip(data_proses.columns[:17], nilai_input)),
                "SAW_Score": round(saw_score, 6),
                "Cluster": cluster_label
            }])

            st.success("✅ Data berhasil diproses!")
            st.markdown(f"### Skor SAW: `{saw_score:.4f}`")
            st.markdown(f"### Segmentasi Keluarga: `Cluster {cluster_label}`")
            st.dataframe(hasil_akhir)

        except Exception as e:
            st.error("Terjadi kesalahan saat memproses data: " + str(e))

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Menu Utama",
        options=["Database", "Proses Segmentasi", "Dashboard", "Inputasi"],
        icons=["folder", "gear", "bar-chart", "pencil"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "black", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "2px"},
            "nav-link-selected": {"background-color": "#4b8bbe"},
        }
    )
    st.session_state.page = selected

if 'page' not in st.session_state:
    st.session_state.page = "Database"

# Routing halaman
if st.session_state.page == "Database":
    page_database()
elif st.session_state.page == "Proses Segmentasi":
    page_segmentasi()
elif st.session_state.page == "Dashboard":
    page_dashboard()
elif st.session_state.page == "Inputasi":
    page_inputasi()
