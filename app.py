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
ZIP_PATH = os.path.join(DATA_DIR, "ecowise_dataset.zip")
EXTRACT_DIR = os.path.join(BASE_DIR, "dataset_temp")
LOGO_PATH = os.path.join(DATA_DIR, "logo_ecowise.png")

os.makedirs(DATA_DIR, exist_ok=True)

# --- LINK DIRECT DOWNLOAD DROPBOX ---
DIRECT_DOWNLOAD_LINK = "https://www.dropbox.com/scl/fi/lc1jivw881j35pr1o78x1/ecowise_dataset.zip?rlkey=7c6b5bk9wwaoeyn2vzbtryo9l&st=y7hv1t34&dl=1"

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True) # Menampilkan Logo
    else:
        st.markdown("<h1 style='text-align: center; color: #2E7D32;'>Eco-Wise</h1>", unsafe_allow_html=True)
    
    st.markdown("### **EcoWise Analysis Dashboard**")
    st.write("""Dashboard ini mempresentasikan insight mengenai dataset klasifikasi sampah yang digunakan dalam pengembangan model AI EcoWise. 
             Melalui visualisasi ini, kita dapat memantau distribusi material limbah guna mendukung sistem pengelolaan sampah cerdas.""")
    st.divider()

# --- FUNGSI DOWNLOAD & EKSTRAK AUTOMATIC ---
@st.cache_resource
def load_and_extract_from_cloud():
    if not os.path.exists(EXTRACT_DIR):
        if not os.path.exists(ZIP_PATH):
            try:
                with st.spinner("📥 Mengunduh dataset asli dari Dropbox Cloud (1.37 GB)... Proses ini memakan waktu beberapa menit."):
                    opener = urllib.request.build_opener() # Atur User-Agent agar unduhan stabil
                    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                    urllib.request.install_opener(opener)
                    
                    urllib.request.urlretrieve(DIRECT_DOWNLOAD_LINK, ZIP_PATH) # Proses pengunduhan langsung ke ZIP_PATH
                st.success("✅ Unduhan zip dari Dropbox selesai!")
            except Exception as e:
                return f"❌ Gagal mengunduh dari Cloud: {str(e)}"
        
        try:
            with st.spinner("🔓 Mengekstrak berkas data limbah..."):
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(EXTRACT_DIR)
            return "✅ Dataset sukses diekstrak & siap!"
        except Exception as e:
            if os.path.exists(ZIP_PATH): # Jika ekstraksi gagal karena berkas zip korup sisa proses sebelumnya, bersihkan berkasnya
                os.remove(ZIP_PATH)
            return f"❌ Gagal mengekstrak berkas: {str(e)}. File rusak telah dibersihkan secara otomatis."
    return "✅ Dataset siap dianalisis."

status_cloud = load_and_extract_from_cloud()
st.sidebar.info(status_cloud)

# --- FUNGSI SCANNING DATASET ---
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

    tab1, tab2, tab3 = st.tabs(["📊 Analisis Data Utama", "🖼️ Sampel Gambar", "📊 Distribusi Material"])
    with tab1:
        # --- VISUALISASI EDA ---
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
            st.markdown(f"**Insight:** Kategori **{main_counts.idxmax()}** adalah yang paling dominan dengan jumlah **{main_counts.max()}** gambar.")
            
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
            
            st.write("**Jumlah Gambar per Kelas (Anorganik, B3, Organik):**")
            for kelas, jumlah in label_counts_main_eda.items():
                st.markdown(f"- &nbsp;&nbsp;**{kelas}**: {jumlah} gambar")

    # --- MATRIKS SUBPLOTS SAMPEL DARI IPYNB ---
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
                if num_samples_to_display == 1 and num_categories_to_plot > 1:
                    axs = axs.reshape(-1, 1)
                elif num_samples_to_display == 1 and num_categories_to_plot == 1:
                    axs = np.array([[axs]])

                for i, category_name in enumerate(sorted_categories_plot):
                    image_paths_list = all_images_by_category_plot[category_name]

                    if image_paths_list:
                        selected_image_paths = np.random.choice(
                            image_paths_list,
                            min(num_samples_to_display, len(image_paths_list)),
                            replace=False
                        )

                        if num_samples_to_display > 0:
                            axs[i, 0].set_ylabel(
                                category_name,
                                rotation=0,
                                size='medium',
                                labelpad=60, 
                                ha='right',
                                va='center',
                                fontweight='bold'
                            )

                        for j in range(num_samples_to_display):
                            if j < len(selected_image_paths):
                                img_path_plot = selected_image_paths[j]
                                try:
                                    img_plot = Image.open(img_path_plot).convert('RGB')
                                    axs[i, j].imshow(img_plot)
                                except Exception:
                                    axs[i, j].set_title("Error", fontsize=8)
                            else:
                                axs[i, j].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)

                            axs[i, j].set_xticks([])
                            axs[i, j].set_yticks([])
                            axs[i, j].set_xlabel('')
                    else:
                        if num_samples_to_display > 0:
                            axs[i, 0].set_ylabel(category_name, rotation=0, size='medium', labelpad=60, ha='right', va='center')
                        for j in range(num_samples_to_display):
                            axs[i, j].text(0.5, 0.5, 'No Images', ha='center', va='center', fontsize=10)
                            axs[i, j].set_xticks([])
                            axs[i, j].set_yticks([])
                            axs[i, j].set_xlabel('')

                plt.subplots_adjust(hspace=0.5, wspace=0.1, top=0.98, left=0.25)
                st.pyplot(fig3)
        else:
            st.warning("Tidak ada data gambar untuk sampel EDA.")

    # --- DISTRIBUSI JENIS MATERIAL SPESIFIK ---
    with tab3:
        st.subheader("📊 Distribusi Semua Jenis Sub-Kategori Material")
        st.write("Grafik di bawah menunjukkan sebaran jumlah data gambar pada seluruh 19 jenis material spesifik secara mendetail.")
        
        fig4, ax4 = plt.subplots(figsize=(12, 8)) # Inisialisasi figure dengan ukuran sesuai skrip ipynb (12, 8)
        
        initial_counts = df['labels_initial'].value_counts().sort_values(ascending=True)  # Mengurutkan agar visualisasi rapi (ascending=True untuk horizontal bar agar yang terbesar di atas)
        
        initial_counts.plot(kind='barh', color='skyblue', ax=ax4) # Membuat plot batang horizontal sesuai spesifikasi skrip asli Anda
        
        ax4.set_title('Distribusi Jenis Material Spesifik (All Categories)', fontsize=14) # Menambahkan konfigurasi teks judul dan label sumbu
        ax4.set_xlabel('Jumlah Gambar')
        ax4.set_ylabel('Material (Kategori - Subfolder)')
        
        for i, v in enumerate(initial_counts.values): # Menambahkan label angka jumlah di samping setiap bar horizontal untuk keterbacaan yang lebih baik di dashboard
            ax4.text(v + (max(initial_counts.values) * 0.005), i, str(v), color='black', va='center', fontweight='bold', fontsize=10)
            
        plt.tight_layout()
        
        st.pyplot(fig4) # Merender grafis ke Tab 3 Streamlit

        st.markdown("- Distribusi jumlah gambar untuk setiap jenis material spesifik. Data diurutkan berdasarkan jumlah gambar secara menaik.")
        st.markdown("- Material Anorganik seperti PET, rag, plastic shopping bags, B3 seperti battery, Organik seperti eggshells, daun, kardus, " \
        "tissue memiliki jumlah gambar yang tinggi, mendekati atau memiliki jumlah gambar yang tinggi mencapai 1000. Ini menunjukkan bahwa dataset " \
        "memiliki representasi yang kuat untuk jenis-jenis material ini.")
        st.markdown("- Material seperti Anorganik HDPEM memiliki jumlah gambar yang relatif lebih rendah yaitu 451 gambar dibandingkan yang lain.")
        st.markdown("- Visualisasi ini membantu mengidentifikasi potensi ketidakseimbangan kelas pada tingkat jenis material yang lebih detail, " \
        "yang penting untuk dipertimbangkan dalam fase pelatihan model nanti. Beberapa jenis B3 (medicine_bottle, aerosol_cans, plastic_detergent_bottles, " \
        "glass_cosmetic_containers, tablet_capsule) memiliki jumlah gambar yang sama yaitu 500, yang lebih sedikit dari kategori lain yang berjumlah sekitar 1000.")

    st.divider()


else:
    st.warning("⚠️ Data kerangka grafik belum siap. Pastikan proses unduh dari Cloud Storage selesai sepenuhnya.")