import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import streamlit as st

st.header("Analisis Pola Pembelian Obat Menggunakan Algoritma FP-Growth")

# Pilih path
path_option = st.selectbox(
    "Pilih Musim",
    ("Musim Kemarau", "Musim Hujan")
)

if path_option == "Musim Kemarau":
    path = "./Data_Obat_Musim_Kemarau.csv"
else:
    path = "./Data_Obat_Musim_Hujan.csv"

# Slider untuk min_support dan min_confidence
min_support = st.slider(
    "Minimal Support (%)",
    1, 100,
    20
) / 100  # Mengubah nilai slider dari persen ke desimal

min_confidence = st.slider(
    "Minimal Confidence (%)",
    1, 100,
    75
) / 100  # Mengubah nilai slider dari persen ke desimal

# Tombol untuk memuat data
if st.button("Tampilkan Hasil"):
    # Load the CSV file
    dirty_df = pd.read_csv(path)

    # Clean the data
    clean_df = dirty_df.drop(118210)
    clean_df = dirty_df.fillna(method='ffill')

    def remove_patterns(text):
        pattern1 = r'B\d+'
        pattern2 = r'C\d+'
        pattern3 = r'D\d+'
        pattern4 = r'E\d+'
        pattern5 = r'TAB \d+ MG'
        pattern6 = r'\d+ MG'

        text = re.sub(pattern1, '', text)
        text = re.sub(pattern2, '', text)
        text = re.sub(pattern3, '', text)
        text = re.sub(pattern4, '', text)
        text = re.sub(pattern5, '', text)
        text = re.sub(pattern6, '', text)
        return text

    clean_df['Nama Obat'] = clean_df['Nama Obat'].apply(remove_patterns)

    rows_with_subtotal = clean_df[clean_df['Nama Obat'].str.contains('Subtotal :')]
    indeks_subtotal = rows_with_subtotal.index
    clean_df.drop(indeks_subtotal, inplace=True)

    clean_df['Nama Obat'] = clean_df['Nama Obat'].str.strip()

    anestesi_alkes_infus_musim_kemarau = [
        "DERMAFIX-T 10 x 25 CM", "I.V CATHETER 18", "SPOIT 1 CC", "SPOIT 5 CC", "SPOIT 10 CC","Total"
        "SUCTION CATHETER 6", "THREE WAY STOPCOCK", "TRANSFUSI SET", "UMBILICAL CORD CLAMP",
        "UNDERPAD", "I.V CATHETER 22", "SPOIT 20 CC", "SPOIT 50 CC", "SPOIT 50 CC CATHETER TIP",
        "I.V CATHETER 24", "DISPOSIBLE NEEDLE", "NYLON 5/0 TS-12 45 CM", "NYLON 3/0 TS-24 75 CM",
        "NYLON 4/0 TS-19 75 CM", "EXTENSION TUBE 200 CM (PERFUSOR)", "ECG ELEKTRODA", "URINE BAG",
        "FOLEY CATHETER 2 WAY 14", "BISTURI 22", "HANDSCOON GYNAECOLOGY SMALL", "HANDSCOON NON STERIL S",
        "HANDSCOON NON STERIL M", "HANDSCOON STERIL 6", "HANDSCOON STERIL 7", "MASKER O2 NON-REB ANAK",
        "MASKER NEBULIZER DEWASA", "FEEDING TUBE NO.8 40 CM", "COLOSTOMY BAG", "ETT 6.5", "ETT 7.0",
        "ETT 5.0", "ETT 4.0", "ETT 4.5", "ETT 5.5", "ETT 6.0", "ETT 7.5", "SUCTION CATHETER 16",
        "SUCTION CATHETER 10 (HD)", "SUCTION CATHETER 12", "SUCTION CATHETER 18", "FOLEY CATHETER 2 WAY 10",
        "GELANG BIRU ANAK", "GELANG PINK ANAK", "SUCTION CATHETER 14", "INFUS SET MICRO (FESCO)",
        "LUMBO SACRAL ORTHOSIS CORSET M", "LUMBO SACRAL ORTHOSIS CORSET L", "LUMBO SACRAL ORTHOSIS CORSET XL",
        "LUMBO SACRAL ORTHOSIS CORSET XXL", "SPINOCAN G.25", "SPINOCAN G.26", "ALAT KATETER",
        "INFUS SET", "CATHETER", "IV SET", "STETOSKOP", "TENSIMETER", "NEBULIZER", "SPIROMETER",
        "SURGICAL MASK", "SURGICAL GLOVES", "SYRINGE", "SCALPEL", "SUTURE MATERIAL", "BLOOD PRESSURE MONITOR",
        "SPOIT", "I.V CATHETER", "ONDANSETRON", "I.V CATHETER 20", "SPOIT 3 CC", "SPOIT 10 CC",
        "OTSU AQUADEST 25 ML (OTSU WI)", "I.V CATHETER 20"

        "METRONIDAZOLE INFUS /ML", "RINGER LACTATE (RL) 500 ML", "NACL 100 ML OTSU (PIGGY BAG)",
        "NOREPHINEPHRINE BITARTRAT INJ", "OTSU DEXTROSE 40% 25 ML", "NACL 100 ML OTSU (PIGGY BAG)",
        "OTSU 500 ML INF", "OTSU ( GLUKOSA) 500 ML INF", "B-FLUID", "GELOFUSIN INFUS",
        "LEVOFLOXACIN (INFUS)", "OTSU KA-EN 3B 500 ML", "ALBUNORM 20% 50 ML", "NEPHROSTERIL INF",
        "OTSU ( GLUKOSA) 500 ML INF", "LEVOFLOXACIN (INFUS)", "OTSU SALIN 3%", "GLUCOSE INFUS",
        "NACL INFUS", "DEXTROSE INFUS", "PARENTERAL NUTRITION", "LIPID INFUS", "VITAMIN INFUS",
        "MINERAL INFUS", "INFUS SET", "AQUADEST", "INFUS SET MACRO (HD)"
    ]

    clean_df = clean_df[~clean_df['Nama Obat'].isin(anestesi_alkes_infus_musim_kemarau)]

    clean_df.reset_index(drop=True, inplace=True)

    fix_df = clean_df.drop([' No.', 'No.RM', 'Embalase', 'Tuslah', 'Total', 'Jml','Biaya'], axis=1)
    le = LabelEncoder()
    fix_df['ID Pasien'] = le.fit_transform(fix_df['Nama Pasien'])

    obat_list = fix_df['Nama Obat'].unique()

    result_df = fix_df[['Tanggal', 'Nama Pasien', 'ID Pasien']].copy()

    for obat in obat_list:
        result_df[obat] = 0

    for idx, row in fix_df.iterrows():
        obat = row['Nama Obat']
        result_df.loc[idx, obat] +=1

    result_df = result_df.drop(columns=['Nama Pasien'])

    grouped_df = result_df.groupby(['Tanggal', 'ID Pasien']).sum()

    grouped_df.reset_index(inplace=True)
    pd.set_option('display.max_columns', None)

    grouped_df['Total'] = grouped_df.iloc[:, 3:].sum(axis=1)

    grouped_df = grouped_df[grouped_df['Total'] >= 10]

    grouped_df.reset_index(drop=True, inplace=True)

    grouped_df.drop(['Total'], axis=1, inplace=True)

    binary_df = grouped_df.drop(columns=['Tanggal', 'ID Pasien'])
    binary_df = binary_df.astype(bool)

    frequent_itemsets = fpgrowth(binary_df, min_support=min_support, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Menampilkan hasil di Streamlit
    st.subheader("Hasil Analisis Aturan Asosiasi dengan Metode FP-Growth")
    st.dataframe(rules)
