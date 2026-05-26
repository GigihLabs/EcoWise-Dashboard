import streamlit as st
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="EcoWise Dashboard", layout="wide", page_icon="🌱")

# --- PATH & DATA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ZIP_PATH = os.path.join(DATA_DIR, "ecowise_dataset_converted.zip")
EXTRACT_DIR = os.path.join(BASE_DIR, "dataset_temp")
LOGO_PATH = os.path.join(DATA_DIR, "logo_ecowise.png")
MANIFEST_PATH = os.path.join(BASE_DIR, "dataset_manifest.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# --- LINK DIRECT DOWNLOAD DROPBOX ---
DIRECT_DOWNLOAD_LINK = "https://www.dropbox.com/scl/fi/uuv7u2442ih0akn4bdd1n/ecowise_dataset_converted.zip?rlkey=bmjmkai89o9nvj91ghxzhh8wz&st=ffqmyeou&dl=1"

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.markdown("<h1 style='text-align: center; color: #2E7D32;'>Eco-Wise</h1>", unsafe_allow_html=True)
    
    st.subheader("EcoWise Analysis Dashboard")
    st.write("""Dashboard ini mempresentasikan insight mengenai dataset klasifikasi sampah yang digunakan dalam pengembangan model AI EcoWise.""")
    st.write("")
    st.write("""Melalui visualisasi ini, kita dapat memantau distribusi material limbah guna mendukung sistem pengelolaan sampah cerdas.""")
    st.divider()

# --- FUNGSI DOWNLOAD & EKSTRAK AUTOMATIC ---
@st.cache_resource
def load_and_extract_from_cloud():
    if not os.path.exists(EXTRACT_DIR):
        if not os.path.exists(ZIP_PATH):
            try:
                with st.spinner("📥 Mengunduh dataset asli dari Dropbox Cloud (1.37 GB)... Proses ini memakan waktu beberapa menit."):
                    opener = urllib.request.build_opener()
                    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                    urllib.request.install_opener(opener)
                    urllib.request.urlretrieve(DIRECT_DOWNLOAD_LINK, ZIP_PATH)
                st.success("✅ Unduhan zip dari Dropbox selesai!")
            except Exception as e:
                return f"❌ Gagal mengunduh dari Cloud: {str(e)}"
        
        try:
            with st.spinner("🔓 Mengekstrak berkas data limbah..."):
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(EXTRACT_DIR)
            return "✅ Dataset sukses diekstrak & siap!"
        except Exception as e:
            if os.path.exists(ZIP_PATH):
                os.remove(ZIP_PATH)
            return f"❌ Gagal mengekstrak berkas: {str(e)}. File rusak telah dibersihkan secara otomatis."
    return "✅ Dataset siap dianalisis."

status_cloud = load_and_extract_from_cloud()
st.sidebar.info(status_cloud)

# =========================================================================
# PROSES LOADING DATASET SECARA OTOMATIS BERBASIS MANIFEST CSV
# =========================================================================
df = pd.DataFrame()

if os.path.exists(MANIFEST_PATH):
    df_manifest = pd.read_csv(MANIFEST_PATH)
    df = df_manifest.copy()
    
    if 'labels_initial' not in df.columns and 'kategori_utama' in df.columns and 'sub_kategori' in df.columns:
        df['labels_initial'] = df['kategori_utama'] + " - " + df['sub_kategori']
        
    if 'brightness' not in df.columns:
        df['brightness'] = 127.0
        
    st.sidebar.success("📊 Sukses memuat data melalui Manifest CSV!")
else:
    st.sidebar.error("❌ Berkas dataset_manifest.csv tidak ditemukan di root direktori repositori!")

# --- MAIN DASHBOARD LAYOUT ---
st.header("🌱 EcoWise Cloud Analysis Dashboard")
st.divider()

if not df.empty:
    # --- METRIK UTAMA ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Gambar", f"{len(df)} foto")
    with col2:
        st.metric("Kategori Utama", f"{df['kategori_utama'].nunique()} Kelas")
    with col3:
        st.metric("Sub-Kategori (Material)", f"{df['sub_kategori'].nunique()} Jenis")

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analisis Data Utama", 
        "🖼️ Sampel Gambar", 
        "📊 Distribusi Material", 
        "📐 Analisis Dimensi Gambar",
        "💡 Analisis Kecerahan Citra"
    ])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Proporsi Sampah")
            fig, ax = plt.subplots(figsize=(10, 6))
            main_counts = df['kategori_utama'].value_counts()
            colors = sns.color_palette('pastel')[0:4]
            ax.pie(main_counts, labels=main_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
            ax.set_title('Proporsi Kategori Utama Sampah', fontsize=14)
            ax.set_ylabel('')  
            st.pyplot(fig)
            
        with c2:
            st.subheader("📈 Distribusi Gambar per Kelas")
            sns.set_style("darkgrid")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            label_counts_main_eda = df['kategori_utama'].value_counts().sort_values(ascending=False)
            
            sns.countplot(x='kategori_utama', data=df, order=label_counts_main_eda.index, palette="viridis", ax=ax2)
            ax2.set_title('Distribusi Gambar per Kelas (Anorganik, B3, Organik)', fontsize=14)
            ax2.set_xlabel('Kelas', fontsize=12)
            ax2.set_ylabel('Jumlah Gambar', fontsize=12)
            
            for i, v in enumerate(label_counts_main_eda.values):
                ax2.text(i, v + (max(label_counts_main_eda.values) * 0.02), str(v), color='black', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig2)

            info_text = f"**Jumlah Gambar per Kelas (Anorganik, B3, Organik):**\n"
            for kelas, jumlah in label_counts_main_eda.items():
                info_text += f"- **{kelas}**: {jumlah} gambar\n"
            st.info(info_text)

    with tab2:
        st.subheader("🖼️ Grid Sampah Berdasarkan Sub-Kategori")
        st.write("Menampilkan 5 sampel gambar secara acak untuk setiap kombinasi Kategori Utama dan Sub-Kategori Material.")
        
        num_samples_to_display = 5
        all_images_by_category_plot = {}

        for (main_cat, sub_cat), group in df.groupby(['kategori_utama', 'sub_kategori']):
            key_name = f"{main_cat}\n({sub_cat})"
            all_images_by_category_plot[key_name] = group['file_path'].tolist()

        sorted_categories_plot = sorted(all_images_by_category_plot.keys())
        num_categories_to_plot = len(sorted_categories_plot)

        if num_categories_to_plot > 0:
            with st.spinner('Membuat grid matriks sampel gambar...'):
                fig3, axs = plt.subplots(
                    num_categories_to_plot, num_samples_to_display,
                    figsize=(num_samples_to_display * 2.5, num_categories_to_plot * 2.5)
                )

                if num_categories_to_plot == 1:
                    axs = np.array([axs])

                for i, category_name in enumerate(sorted_categories_plot):
                    image_paths_list = all_images_by_category_plot[category_name]

                    if image_paths_list:
                        cleaned_paths = []
                        for p in image_paths_list:
                            p_str = str(p)
                            if "/content/drive/MyDrive/EcoWise/dataset-trashcan/" in p_str:
                                local_p = p_str.replace("/content/drive/MyDrive/EcoWise/dataset-trashcan/", f"{EXTRACT_DIR}/dataset-trashcan/")
                                cleaned_paths.append(local_p)
                            else:
                                cleaned_paths.append(p_str)

                        selected_image_paths = np.random.choice(
                            cleaned_paths,
                            min(num_samples_to_display, len(cleaned_paths)),
                            replace=False
                        )

                        axs[i, 0].set_ylabel(
                            category_name, rotation=0, size='medium',
                            labelpad=60, ha='right', va='center', fontweight='bold'
                        )

                        for j in range(num_samples_to_display):
                            if j < len(selected_image_paths):
                                img_path_plot = selected_image_paths[j]
                                try:
                                    img_plot = Image.open(img_path_plot).convert('RGB')
                                    axs[i, j].imshow(img_plot)
                                except Exception:
                                    axs[i, j].text(0.5, 0.5, 'Error\nImage', ha='center', va='center', fontsize=8)
                            else:
                                axs[i, j].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)

                            axs[i, j].set_xticks([])
                            axs[i, j].set_yticks([])
                    else:
                        axs[i, 0].set_ylabel(category_name, rotation=0, size='medium', labelpad=60, ha='right', va='center')
                        for j in range(num_samples_to_display):
                            axs[i, j].text(0.5, 0.5, 'No Images', ha='center', va='center', fontsize=10)
                            axs[i, j].set_xticks([])
                            axs[i, j].set_yticks([])

                plt.subplots_adjust(hspace=0.5, wspace=0.1, top=0.98, left=0.25)
                st.pyplot(fig3)

    with tab3:
        st.subheader("📊 Distribusi Semua Jenis Sub-Kategori Material")
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        initial_counts = df['labels_initial'].value_counts().sort_values(ascending=True)
        initial_counts.plot(kind='barh', color='#4682B4', ax=ax4)
        ax4.set_title('Distribusi Jenis Material Spesifik (All Categories)', fontsize=14)
        for i, v in enumerate(initial_counts.values):
            ax4.text(v + (max(initial_counts.values) * 0.005), i, str(v), color='black', va='center', fontweight='bold', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig4)

    with tab4:
        st.subheader("📐 Distribusi Aspect Ratio Dataset Citra EcoWise")
        valid_sizes_df = df[(df['width'] > 0) & (df['height'] > 0)]
        if not valid_sizes_df.empty:
            ratios = valid_sizes_df['width'] / valid_sizes_df['height']
            fig5, ax5 = plt.subplots(figsize=(10, 5))
            ax5.hist(ratios, bins=20, color='#4682B4', edgecolor='white', alpha=0.7)
            ax5.set_title("Distribusi Aspect Ratio (Width / Height)", fontsize=14, fontweight='bold')
            st.pyplot(fig5)

    with tab5:
        st.subheader("💡 Distribusi Kecerahan (Brightness) Dataset Citra EcoWise")
        if os.path.exists(os.path.join(BASE_DIR, "hist_brightness.png")):
            st.image(os.path.join(BASE_DIR, "hist_brightness.png"), use_container_width=True, caption="Histogram Tingkat Kecerahan Pra-Kalkulasi")
        else:
            st.warning("⚠️ Berkas gambar 'hist_brightness.png' belum diunggah ke root repositori GitHub.")