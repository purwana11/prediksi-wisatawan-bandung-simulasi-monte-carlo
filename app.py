# app.py

import pandas as pd
import numpy as np
import random
from flask import Flask, render_template, request

app = Flask(__name__)

# --- 1. Definisi dan Pendaftaran Filter Jinja2 ---
def format_ribuan(value):
    """Fungsi untuk memformat angka menjadi format ribuan (misal: 1,000,000)."""
    try:
        # Menggunakan format_string Python f"{value:,}"
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return value

# Daftarkan Filter Kustom ke Jinja Environment
app.jinja_env.filters['format_ribuan'] = format_ribuan


# --- 2. Logika Inti Simulasi Monte Carlo ---
def run_monte_carlo_prediction(NUM_SIMULATIONS):
    # Nama file dataset yang diunggah
    file_name = "disparbud-od_15367_jml_pengunjung_ke_objek_wisata__jenis_wisatawan_ka_v2_data.csv"
    
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        return {"error": "File data tidak ditemukan. Pastikan 'disparbud-od_15367_jml_pengunjung_ke_objek_wisata__jenis_wisatawan_ka_v2_data.csv' ada di direktori yang sama."}

    # Data Preparation: Filter dan Agregasi untuk area 'Bandung'
    bandung_df = df[df['nama_kabupaten_kota'].str.contains('BANDUNG', case=False, na=False)].copy()
    yearly_visitors = bandung_df.groupby('tahun')['jumlah_pengunjung'].sum().reset_index()
    yearly_visitors.columns = ['Tahun', 'Jumlah Pengunjung']
    
    visitors_df = yearly_visitors.copy()
    total_visitors = visitors_df['Jumlah Pengunjung'].sum()
    
    # 3. Perhitungan Probabilitas dan Interval (0-999)
    visitors_df['Probabilitas'] = visitors_df['Jumlah Pengunjung'] / total_visitors
    visitors_df['Kumulatif'] = visitors_df['Probabilitas'].cumsum()
    
    visitors_df['Batas Bawah'] = (visitors_df['Kumulatif'].shift(1, fill_value=0) * 1000).round().astype(int)
    visitors_df['Batas Atas'] = (visitors_df['Kumulatif'] * 1000).round().astype(int) - 1
    
    visitors_df.loc[visitors_df.index[0], 'Batas Bawah'] = 0
    visitors_df.loc[visitors_df.index[-1], 'Batas Atas'] = 999
    
    # Format data untuk tabel interval yang dikirim ke HTML
    interval_data = visitors_df[['Tahun', 'Jumlah Pengunjung', 'Probabilitas', 'Kumulatif', 'Batas Bawah', 'Batas Atas']].copy()
    interval_data['Probabilitas'] = interval_data['Probabilitas'].map('{:.4f}'.format)
    interval_data['Kumulatif'] = interval_data['Kumulatif'].map('{:.4f}'.format)
    interval_data['Interval'] = interval_data.apply(lambda row: f"{row['Batas Bawah']:03d} - {row['Batas Atas']:03d}", axis=1)

    # 4. Fungsi Pemetaan Prediksi
    def get_prediction(rand_num, df_int):
        row = df_int[(df_int['Batas Bawah'] <= rand_num) & (df_int['Batas Atas'] >= rand_num)]
        return row['Jumlah Pengunjung'].iloc[0] if not row.empty else np.nan

    # 5. Eksekusi Simulasi
    # Menggunakan seed tetap agar prediksi pada jumlah N yang sama selalu konsisten
    random.seed(42) 
    random_numbers = [random.randint(0, 999) for _ in range(NUM_SIMULATIONS)]
    
    simulation_data = []
    predicted_values = []

    for i, rn in enumerate(random_numbers):
        prediksi = get_prediction(rn, visitors_df)
        if pd.notna(prediksi):
            predicted_values.append(int(prediksi))
        
        simulation_data.append({
            'No.': i + 1,
            'Angka Acak': f"{rn:03d}",
            # Kirim nilai integer untuk format_ribuan di template
            'Prediksi': int(prediksi) if pd.notna(prediksi) else 0 
        })

    # 6. Hitung Hasil Akhir
    final_prediction = int(round(np.mean(predicted_values))) if predicted_values else 0

    return {
        'interval_table': interval_data.to_dict('records'),
        'simulation_results': simulation_data,
        'final_prediction': final_prediction,
        'num_simulations': NUM_SIMULATIONS
    }

# --- 3. Rute Flask (GET/POST) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    default_sims = 5
    
    if request.method == 'POST':
        # Ambil input jumlah simulasi dari form
        try:
            num_sims = int(request.form.get('num_simulations', default_sims))
            # Batasi minimal 1 dan maksimal 1000 untuk performa
            num_sims = max(1, min(num_sims, 1000))
        except ValueError:
            num_sims = default_sims
    else:
        # Permintaan GET awal
        num_sims = default_sims
        
    results = run_monte_carlo_prediction(num_sims)
    
    if 'error' in results:
        # Tampilkan error jika data tidak ditemukan
        return f"<h1>Error Data:</h1><p>{results['error']}</p>"
        
    return render_template('index.html', results=results)

if __name__ == '__main__':
    # Pastikan Anda menginstal Flask, pandas, dan numpy
    app.run(debug=True)