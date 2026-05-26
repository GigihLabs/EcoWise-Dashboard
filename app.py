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

# KUNCI PERBAIKAN: Ubah dari EXTRACT_DIR ke BASE_DIR
MANIFEST_PATH = os.path.join(BASE_DIR, "dataset_manifest.csv") 

os.makedirs(DATA_DIR, exist_ok=True)

# --- LINK DIRECT DOWNLOAD DROPBOX ---
DIRECT_DOWNLOAD_LINK = "https://www.dropbox.com/scl/fi/uuv7u2442ih0akn4bdd1n/ecowise_dataset_converted.zip?rlkey=bmjmkai89o9nvj91ghxzhh8wz&st=ffqmyeou&dl=1"

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True) # Menampilkan Logo
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

# --- FUNGSI SCANNING DATASET SECARA MANUAL (FALLBACK JIKA MANIFEST HILANG) ---
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
                        full_path = os.path.join(root, file)
                        width, height = 0, 0
                        brightness_val = 0.0
                        try:
                            with Image.open(full_path) as img:
                                width, height = img.size
                                img_gray = img.convert("L")
                                brightness_val = float(np.mean(img_gray))
                        except Exception:
                            pass
                        
                        file_data.append({
                            "file_path": full_path,
                            "kategori_utama": main_cat, 
                            "sub_kategori": sub_cat,
                            "labels_initial": f"{main_cat} - {sub_cat}",
                            "width": width,
                            "height": height,
                            "brightness": brightness_val
                        })
    return pd.DataFrame(file_data)

# =========================================================================
# PROSES LOADING DATASET SECARA OTOMATIS BERBASIS MANIFEST CSV (LOCKED)
# =========================================================================
df = pd.DataFrame()

# Kita kunci pembacaan data MUTLAK hanya dari Manifest CSV agar data tidak duplikat
if os.path.exists(MANIFEST_PATH):
    df = pd.read_csv(MANIFEST_PATH)
    
    # Penyelarasan kolom
    if 'labels_initial' not in df.columns and 'kategori_utama' in df.columns and 'sub_kategori' in df.columns:
        df['labels_initial'] = df['kategori_utama'] + " - " + df['sub_kategori']
        
    if 'brightness' not in df.columns:
        df['brightness'] = 127.0
        
    st.sidebar.success("📊 Sukses memuat data melalui Manifest CSV!")
else:
    st.sidebar.error("❌ Berkas dataset_manifest.csv tidak ditemukan di root repositori!")

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
        # --- VISUALISASI EDA ---
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
            
            plt.tight_layout()
            st.pyplot(fig2)

            info_text = f"**Jumlah Gambar per Kelas (Anorganik, B3, Organik):**\n"
            for kelas, jumlah in label_counts_main_eda.items():
                info_text += f"- **{kelas}**: {jumlah} gambar\n"
            st.info(info_text)

        st.info(f"""
                💡 **Insight Analisis Data Utama:**
                * Kategori gambar sampah 'Anorganik' dan 'Organik' adalah yang paling dominan, yaitu sebanyak 6.471 gambar sampah 'Anorganik' diikuti oleh 'Organik' sebanyak 6.000 gambar.
                * Terdapat ketidakseimbangan yang signifikan antar kelas. Kategori 'Non-Waste' memiliki jumlah gambar yang paling sedikit (1.649 gambar), menjadikannya kelas minoritas dibandingkan dengan kategori 'Anorganik' dan 'Organik' yang jauh lebih banyak. Kategori 'B3' juga tergolong minoritas dengan 3.500 gambar.
                """)

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
                        # Substitusi lokasi path agar gambar tetap terbaca dari folder ekstraksi lokal
                        cleaned_paths = []
                        for p in image_paths_list:
                            p_str = str(p)
                            if "/content/drive/MyDrive/EcoWise/dataset-trashcan/" in p_str:
                                local_p = p_str.replace("/content/drive/MyDrive/EcoWise/dataset-trashcan/", f"{EXTRACT_DIR}/dataset-trashcan/")
                                cleaned_paths.append(local_p)
                            elif "dataset_temp" in p_str:
                                cleaned_paths.append(p_str)
                            else:
                                # Fallback alternatif konstruksi path manual
                                normalized_path = os.path.join(EXTRACT_DIR, "dataset-trashcan", category_name.split('\n')[0], category_name.split('\n')[1].replace('(', '').replace(')', ''), os.path.basename(p_str))
                                cleaned_paths.append(normalized_path)

                        selected_image_paths = np.random.choice(
                            cleaned_paths,
                            min(num_samples_to_display, len(cleaned_paths)),
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
                                    axs[i, j].set_title("N/A (Path Error)", fontsize=8)
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

        # =========================================================================
        # PERBAIKAN DINAMIS UNTUK MEMUAT GAMBAR DARI EXTRACT_DIR
        # =========================================================================
        raw_path = row['file_path'] # Nilai asli: /content/drive/MyDrive/.../Anorganik/kaleng/xyz.jpg
        
        # 1. Cari kata kunci 'Anorganik', 'Organik', atau 'B3' sebagai titik potong folder asli
        parsed_path = ""
        for category in ["Anorganik", "Organik", "B3", "Non-Waste"]:
            if category in raw_path:
                # Ambil jalur dari nama kategori ke belakang (misal: Anorganik/kaleng/xyz.jpg)
                parsed_path = raw_path[raw_path.find(category):]
                break
            
        # 2. Gabungkan dengan direktori ekstraksi lokal server (EXTRACT_DIR)
        if parsed_path:
            # Jika struktur zip Anda langsung berisi kategori, gunakan ini:
            img_path = os.path.join(EXTRACT_DIR, parsed_path)
            
            # JIKA di dalam zip Anda ternyata ada bungkus folder lagi bernama 'dataset-trashcan', gunakan ini:
            if not os.path.exists(img_path):
                img_path = os.path.join(EXTRACT_DIR, "dataset-trashcan", parsed_path)
                
        else:
            img_path = raw_path
            
        # 3. Alur pemuatan gambar ke Streamlit
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path)
                st.image(image, caption=f"ID: {row['file_id']} | {row['kategori_utama']}", use_container_width=True)
                
            except Exception as e:
                st.error(f"Error membuka gambar: {e}")
                
        else:
            # Log pembantu untuk melihat di mana Streamlit Cloud mencari file tersebut secara fisik
            st.warning(f"⚠️ File fisik tidak ditemukan di server pada lokasi: {img_path}")

        st.info(f"""
                💡 **Insight Sampel Gambar:**
                * Berdasarkan visualisasi sampel gambar, dapat dilihat bahwa dataset memiliki variasi yang cukup baik dalam hal jenis objek, latar belakang, pencahayaan, dan sudut pengambilan gambar di setiap kategori.
                * Variasi ini menunjukkan bahwa dataset memiliki kompleksitas yang cukup tinggi, sehingga model yang digunakan harus mampu melakukan generalisasi dengan baik terhadap berbagai kondisi citra.
                """)

    with tab3:
        st.subheader("📊 Distribusi Semua Jenis Sub-Kategori Material")
        st.write("Grafik di bawah menunjukkan sebaran jumlah data gambar pada seluruh 20 jenis material spesifik secara mendetail.")
        
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        initial_counts = df['labels_initial'].value_counts().sort_values(ascending=True)
        
        initial_counts.plot(kind='barh', color='#4682B4', ax=ax4)
        ax4.set_title('Distribusi Jenis Material Spesifik (All Categories)', fontsize=14)
        ax4.set_xlabel('Jumlah Gambar')
        ax4.set_ylabel('Material (Kategori - Subfolder)')
        
        for i, v in enumerate(initial_counts.values):
            ax4.text(v + (max(initial_counts.values) * 0.005), i, str(v), color='black', va='center', fontweight='bold', fontsize=10)
            
        plt.tight_layout()
        st.pyplot(fig4)

        st.info(f"""
                💡 **Insight Distribusi Material:**
                * Material Anorganik seperti PET, rag, plastic shopping bags, B3 seperti battery, Organik seperti eggshells, daun, kardus memiliki representasi yang kuat.
                * Distribusi jumlah gambar untuk setiap jenis material spesifik. Data diurutkan berdasarkan jumlah gambar secara menaik.
                """)

    with tab4:
        st.subheader("📐 Distribusi Aspect Ratio Dataset Citra EcoWise")
        st.write("Visualisasi histogram di bawah memaparkan sebaran perbandingan lebar terhadap tinggi (*Width / Height*) dari seluruh citra sampah.")

        valid_sizes_df = df[(df['width'] > 0) & (df['height'] > 0)]

        if not valid_sizes_df.empty:
            ratios = valid_sizes_df['width'] / valid_sizes_df['height']

            fig5, ax5 = plt.subplots(figsize=(10, 5))
            ax5.hist(ratios, bins=20, color='#4682B4', edgecolor='white', alpha=0.7)
            ax5.set_title("Distribusi Aspect Ratio (Width / Height)", fontsize=14, fontweight='bold')
            ax5.set_xlabel("Rasio Aspek (Width / Height)", fontsize=11)
            ax5.set_ylabel("Frekuensi / Jumlah Gambar", fontsize=11)
            ax5.grid(axis='y', linestyle='--', alpha=0.5)
            
            st.pyplot(fig5)
            
            ratios_np = np.array(ratios)
            st.info(f"""
            💡 **Insight Analisis Dimensi:**
            * **Nilai Rasio Minimum:** {ratios_np.min():.2f}
            * **Nilai Rasio Maksimum:** {ratios_np.max():.2f}
            * **Rata-rata Rasio (Mean):** {ratios_np.mean():.2f}
            * **Median Rasio:** {np.median(ratios_np):.2f}
            * Mayoritas gambar dalam dataset memiliki rasio aspek mendekati 1.0 (persegi), hal ini terlihat dari nilai median dan rata-rata rasio aspek. Artinya mayoritas gambar memiliki lebar dan tinggi yang sama atau sangat mirip.
            * Variasi Rasio Aspek yang Signifikan, dibuktikan variasi rasio aspek yang cukup luas, mulai dari nilai 0.28 hingga 6.21. Ini berarti dataset memiliki gambar dengan orientasi potret dan lanskap yang ekstrem.
            """)
        else:
            st.warning("⚠️ Tidak ada data ukuran gambar yang valid ditemukan untuk melakukan visualisasi aspek rasio.")

    with tab5:
        st.subheader("💡 Distribusi Kecerahan (Brightness) Dataset Citra EcoWise")
        st.write("Visualisasi di bawah merupakan histogram representasi tingkat kecerahan rata-rata piksel dari citra setelah dikonversi ke skala abu-abu (Grayscale).")
        
        # Daftar jalur pencarian berkas gambar grafik visualisasi statis
        kandidat_path = [
            os.path.join(BASE_DIR, "hist_brightness.png"),
            os.path.join(BASE_DIR, "dfghj.png"),
            os.path.join(DATA_DIR, "hist_brightness.png"),
            os.path.join(DATA_DIR, "dfghj.png")
        ]
        
        gambar_ditemukan = False
        for path in kandidat_path:
            if os.path.exists(path):
                st.image(path, use_container_width=True)
                gambar_ditemukan = True
                break
        
        if not gambar_ditemukan:
            st.warning("⚠️ Berkas gambar grafik ('hist_brightness.png' atau 'dfghj.png') tidak ditemukan di direktori utama maupun di folder data.")
            st.info("""
            💡 **Langkah Penyelesaian:**
            1. Pastikan Anda sudah menyimpan file gambar hasil plot notebook Anda ke dalam folder proyek utama yang sama dengan berkas dashboard ini.
            2. Pastikan nama file ditulis dengan huruf kecil semua (`hist_brightness.png`).
            """)
        
        st.info("""
                💡 **Insight Analisis Kecerahan (Brightness):**
                * **Kecerahan Minimum:** 0.00
                * **Kecerahan Maksimum:** 255.00
                * **Rata-rata Kecerahan:** 169.59
                * **Nilai Tengah Kecerahan:** 169.25
                * **Nilai Maksimum Kecerahan:** 250.85
                * **Nilai Minimum Kecerahan:** 51.97
                * Mayoritas gambar dalam dataset memiliki tingkat kecerahan sedang, karena rata-rata nilai kecerahan hampir sama dengan nilai mediannya. Ini menunjukkan bahwa dataset secara keseluruhan memiliki pencahayaan yang seimbang.
                * Terdapat rentang kecerahan yang cukup luas, dari gambar yang sangat gelap hingga sangat terang. Artinya dataset mencakup gambar yang diambil dalam berbagai kondisi pencahayaan, sehingga baik untuk melatih model agar robust terhadap variasi dunia nyata.
                """)

    st.divider()
else:
    st.warning("⚠️ Data kerangka grafik belum siap. Pastikan proses unduh dari Cloud Storage selesai sepenuhnya.")


# conda activate main-ds
# streamlit run app.py