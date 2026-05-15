import streamlit as st
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="EcoWise Dashboard", layout="wide", page_icon="🌱")

# --- PATH & DATA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "data", "ecowise_dataset.zip")
EXTRACT_DIR = os.path.join(BASE_DIR, "dataset_temp")
LOGO_PATH = os.path.join(BASE_DIR, "data", "logo_ecowise.png")

# --- SIDEBAR (Logo & Penjelasan Proyek) ---
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.markdown("<h1 style='text-align: center; color: #2E7D32;'>Eco-Wise</h1>", unsafe_allow_html=True)
    
    st.markdown("### **EcoWise Analysis Dashboard**")
    st.write("""
    Dashboard ini mempresentasikan insight mengenai dataset klasifikasi sampah yang digunakan dalam pengembangan model AI EcoWise. 
    Melalui visualisasi ini, kita dapat memantau distribusi material limbah guna mendukung sistem pengelolaan sampah cerdas.
    """)
    st.divider()

# --- FUNGSI LOAD & EXTRACTION ---
@st.cache_resource
def load_and_extract():
    if not os.path.exists(EXTRACT_DIR):
        if os.path.exists(ZIP_PATH):
            try:
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(EXTRACT_DIR)
                return "✅ Dataset berhasil dimuat!"
            except Exception as e:
                return f"❌ Gagal mengekstrak: {str(e)}"
        else:
            return f"⚠️ File ZIP tidak ditemukan di folder data/. Menunggu konfigurasi cloud external..."
    return "✅ Dataset siap dianalisis."

st.sidebar.info(load_and_extract())

# --- FUNGSI SCAN DATA ---
@st.cache_data
def scan_data(base_dir):
    file_data = []
    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                rel_path = os.path.relpath(root, base_dir)
                path_parts = rel_path.split(os.sep)
                if path_parts[0] == 'dataset-trashcan' or path_parts[0] == '.':
                    path_parts = path_parts[1:]
                if len(path_parts) >= 2:
                    main_cat = path_parts[0]
                    sub_cat = path_parts[1]
                    for file in files:
                        file_data.append({
                            "file_path": os.path.join(root, file),
                            "kategori_utama": main_cat, 
                            "sub_kategori": sub_cat,
                            "labels_initial": f"{main_cat} - {sub_cat}"
                        })
    return pd.DataFrame(file_data)

df = scan_data(EXTRACT_DIR)

# --- MAIN CONTENT ---
st.header("🌱 Eco-Wise Analysis Dashboard")

if not df.empty:
    tab1, tab2, tab3 = st.tabs([
        "📊 Analisis Data Utama", 
        "🖼️ Sampel Gambar per Sub-Kategori", 
        "📊 Distribusi Material Spesifik"
    ])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Gambar", f"{len(df)} foto")
        with col2:
            st.metric("Kategori Utama", f"{df['kategori_utama'].nunique()} Kelas")
        with col3:
            st.metric("Sub-Kategori (Material)", f"{df['sub_kategori'].nunique()} Jenis")
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Proporsi Sampah")
            fig, ax = plt.subplots(figsize=(10, 6))
            main_counts = df['kategori_utama'].value_counts()
            colors = sns.color_palette('pastel')[0:3]
            ax.pie(main_counts, labels=main_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
            ax.set_title('Proporsi Kategori Utama Limbah (Eco-Wise)', fontsize=14)
            ax.set_ylabel('')  
            st.pyplot(fig)
            st.info(f"**Insight:** Kategori **{main_counts.idxmax()}** adalah yang paling dominan dengan jumlah **{main_counts.max()}** gambar.")
            
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

        st.divider()
        st.subheader("🔍 Eksplorasi Data Acak")
        selected_cat = st.selectbox("Filter berdasarkan Kategori Utama:", df['kategori_utama'].unique())
        filtered_df = df[df['kategori_utama'] == selected_cat]
        sample_size = min(5, len(filtered_df))
        sample_df = filtered_df.sample(n=sample_size)
        img_cols = st.columns(sample_size)
        for i, (_, row) in enumerate(sample_df.iterrows()):
            with img_cols[i]:
                img = Image.open(row['file_path'])
                st.image(img, caption=f"Material: {row['sub_kategori']}", width=150)

    with tab2:
        st.subheader("🖼️ Grid Sampah Berdasarkan Sub-Kategori")
        num_samples_to_display = 5
        all_images_by_category_plot = {}
        for (main_cat, sub_cat), group in df.groupby(['kategori_utama', 'sub_kategori']):
            key_name = f"{main_cat}\n({sub_cat})"
            all_images_by_category_plot[key_name] = group['file_path'].tolist()

        sorted_categories_plot = sorted(all_images_by_category_plot.keys())
        num_categories_to_plot = len(sorted_categories_plot)

        if num_categories_to_plot > 0:
            with st.spinner('Membuat grid matriks sampel gambar...'):
                fig3, axs = plt.subplots(num_categories_to_plot, num_samples_to_display, figsize=(num_samples_to_display * 2.5, num_categories_to_plot * 2.5))
                if num_categories_to_plot == 1: axs = np.array([axs])
                if num_samples_to_display == 1 and num_categories_to_plot > 1: axs = axs.reshape(-1, 1)
                elif num_samples_to_display == 1 and num_categories_to_plot == 1: axs = np.array([[axs]])

                for i, category_name in enumerate(sorted_categories_plot):
                    image_paths_list = all_images_by_category_plot[category_name]
                    if image_paths_list:
                        selected_image_paths = np.random.choice(image_paths_list, min(num_samples_to_display, len(image_paths_list)), replace=False)
                        axs[i, 0].set_ylabel(category_name, rotation=0, size='medium', labelpad=60, ha='right', va='center', fontweight='bold')
                        for j in range(num_samples_to_display):
                            if j < len(selected_image_paths):
                                try:
                                    img_plot = Image.open(selected_image_paths[j]).convert('RGB')
                                    axs[i, j].imshow(img_plot)
                                except Exception:
                                    axs[i, j].set_title("Error", fontsize=8)
                            else:
                                axs[i, j].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
                            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
                    else:
                        axs[i, 0].set_ylabel(category_name, rotation=0, size='medium', labelpad=60, ha='right', va='center')
                        for j in range(num_samples_to_display):
                            axs[i, j].text(0.5, 0.5, 'No Images', ha='center', va='center', fontsize=10)
                            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
                plt.subplots_adjust(hspace=0.5, wspace=0.1, top=0.98, left=0.25)
                st.pyplot(fig3)

    with tab3:
        st.subheader("📊 Distribusi Semua Jenis Sub-Kategori Material")
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        initial_counts = df['labels_initial'].value_counts().sort_values(ascending=True)
        initial_counts.plot(kind='barh', color='skyblue', ax=ax4)
        ax4.set_title('Distribusi Jenis Material Spesifik (All Categories)', fontsize=14)
        ax4.set_xlabel('Jumlah Gambar')
        ax4.set_ylabel('Material (Kategori - Subfolder)')
        for i, v in enumerate(initial_counts.values):
            ax4.text(v + (max(initial_counts.values) * 0.005), i, str(v), color='black', va='center', fontweight='bold', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig4)
else:
    st.info("💡 Hubungkan dataset eksternal atau unggah file ecowise_dataset.zip ke dalam folder data/ untuk melihat visualisasi data.")
