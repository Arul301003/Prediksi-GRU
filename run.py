import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model("model/final_model_gru (14).h5")
st.set_page_config(page_title="Cryptocurrency Price Prediction", layout="wide", page_icon="📈")

# Custom CSS & NAVBAR HTML
st.markdown("""
<style>
    /* Reset */
    body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        line-height: 1.6;
        padding-top: 0 !important;
    }

    /* Navbar */
    .navbar {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        padding: 1rem 2rem;
        border-radius: 0 0 15px 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        position: sticky;
        top: 0;
        z-index: 1000;
    }

    .navbar h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .nav-links {
        display: flex;
        gap: 25px;
    }

    .nav-links a {
        color: white;
        text-decoration: none;
        font-weight: 500;
        padding: 5px 10px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }

    .nav-links a:hover {
        background-color: rgba(255,255,255,0.2);
        transform: translateY(-2px);
    }
    

    /* Section container */
    .section {
        background-color: white;
        margin: 2rem auto;
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        max-width: 1000px;
        border: 1px solid #e2e8f0;
    }

    .section-header {
        color: #2563eb;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .trend-up {
        color: #10b981;
        font-weight: bold;
        font-size: 1.4rem;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    .trend-down {
        color: #ef4444;
        font-weight: bold;
        font-size: 1.4rem;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, rgba(37,99,235,0.1) 0%, rgba(37,99,235,0.5) 50%, rgba(37,99,235,0.1) 100%);
    }

    .feature-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2563eb;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.05);
    }

    .feature-card h3 {
        margin-top: 0;
        color: #1e40af;
    }

    .tech-badge {
        display: inline-block;
        background-color: #e0e7ff;
        color: #2563eb;
        padding: 5px 12px;
        border-radius: 20px;
        margin: 5px;
        font-weight: 500;
        font-size: 0.9rem;
    }

    .input-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 15px;
        margin-bottom: 2rem;
    }

    .input-day {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }

    .input-day label {
        font-weight: 600;
        color: #4b5563;
        margin-bottom: 5px;
        display: block;
    }

    .result-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        border: 1px solid #bfdbfe;
    }

    .result-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e40af;
    }

    .stNumberInput input {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }

    @media (max-width: 768px) {
        .input-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<!-- Navbar -->
<div class="navbar">
    <h1>
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="1" x2="12" y2="3"></line>
            <line x1="12" y1="21" x2="12" y2="23"></line>
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
            <line x1="1" y1="12" x2="3" y2="12"></line>
            <line x1="21" y1="12" x2="23" y2="12"></line>
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
        </svg>
        CryptoPredict AI
    </h1>
    <div class="nav-links">
        <a href="#beranda">Beranda</a>
        <a href="#tentang">Tentang</a>
        <a href="#teknologi">Teknologi</a>
        <a href="#prediksi">Prediksi</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- BERANDA ---
st.markdown('<div id="beranda" class="section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg> Beranda</h2>', unsafe_allow_html=True)
st.write("""
<div style="font-size: 1.1rem;">
    Selamat datang di <strong>CryptoPredict AI</strong> - solusi prediksi harga cryptocurrency berbasis kecerdasan buatan. 
    Aplikasi ini menggunakan model <strong>GRU (Gated Recurrent Unit)</strong> yang canggih untuk menganalisis pergerakan harga dan memberikan prediksi akurat untuk hari berikutnya.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
    <div style="margin-top: 1.5rem;">
        <h3 style="color: #2563eb;">✨ Fitur Unggulan</h3>
        <ul style="padding-left: 1.5rem;">
            <li>Prediksi berbasis machine learning</li>
            <li>Analisis tren visual yang interaktif</li>
            <li>Antarmuka yang intuitif dan mudah digunakan</li>
            <li>Dukungan untuk berbagai cryptocurrency</li>
            <li>Update model secara berkala</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3", caption="Ilustrasi Prediksi Harga", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- TENTANG ---
st.markdown('<div id="tentang" class="section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg> Tentang Aplikasi</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-card">
    <h3>📊 Analisis Time Series</h3>
    <p>Aplikasi ini dirancang khusus untuk menganalisis data time series harga cryptocurrency menggunakan pendekatan deep learning yang canggih.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-card">
    <h3>🧠 Model GRU</h3>
    <p>Menggunakan arsitektur Gated Recurrent Unit (GRU) yang secara khusus efektif untuk memproses data sekuensial seperti pergerakan harga.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-card">
    <h3>👥 Untuk Siapa?</h3>
    <p>
        <strong>Trader & Investor:</strong> Membantu pengambilan keputusan trading<br>
        <strong>Analis Pasar:</strong> Alat bantu analisis teknikal<br>
        <strong>Pencari Ilmu:</strong> Belajar penerapan AI di finansial
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- TEKNOLOGI ---
st.markdown('<div id="teknologi" class="section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"></path><path d="M2 17l10 5 10-5"></path><path d="M2 12l10 5 10-5"></path></svg> Teknologi yang Digunakan</h2>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 2rem;">
    <span class="tech-badge">Python 3</span>
    <span class="tech-badge">TensorFlow</span>
    <span class="tech-badge">Keras</span>
    <span class="tech-badge">Streamlit</span>
    <span class="tech-badge">GRU Networks</span>
    <span class="tech-badge">Matplotlib</span>
    <span class="tech-badge">Scikit-learn</span>
    <span class="tech-badge">NumPy</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style="color: #2563eb;">Arsitektur Model</h3>
<p>Model GRU kami dirancang dengan lapisan-lapisan berikut untuk memproses data time series:</p>
""", unsafe_allow_html=True)

code = """
# Contoh arsitektur model GRU
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(10, 1)),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
# Pelatihan model dengan data historis
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
"""

st.code(code, language='python')

st.markdown('</div>', unsafe_allow_html=True)

# --- PREDIKSI ---
st.markdown('<div id="prediksi" class="section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg> Prediksi Harga</h2>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <p>Masukkan harga <strong>10 hari terakhir</strong> untuk melihat prediksi harga esok hari dan analisis tren:</p>
</div>
""", unsafe_allow_html=True)

# Input harga dalam grid yang lebih baik
st.markdown('<div class="input-grid">', unsafe_allow_html=True)
prices = []
for i in range(10):
    with st.container():
        day_num = 10 - i
        st.markdown(f'<div class="input-day"><label>Hari ke-{day_num}</label>', unsafe_allow_html=True)
        price = st.number_input(
            f"Harga Hari ke-{day_num}", 
            min_value=0.0, 
            step=100.0, 
            format="%.2f", 
            key=f"day{i}",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        prices.append(price)
st.markdown('</div>', unsafe_allow_html=True)

# Proses prediksi jika input lengkap
if len(prices) == 10 and all(p > 0 for p in prices):
    st.markdown('<h3 style="color: #2563eb;">📊 Visualisasi Data</h3>', unsafe_allow_html=True)
    
    # Buat grafik yang lebih profesional
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 11), prices, marker='o', color='#2563eb', linewidth=2, markersize=8)
    ax.fill_between(range(1, 11), prices, color='#2563eb', alpha=0.1)
    ax.set_xlabel("Hari ke-", fontsize=12)
    ax.set_ylabel("Harga (USD)", fontsize=12)
    ax.set_title("Perkembangan Harga 10 Hari Terakhir", fontsize=14, pad=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # Prediksi
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(np.array(prices).reshape(-1, 1))
    input_data = np.reshape(scaled, (1, 10, 1))
    prediction = model.predict(input_data)
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    current_price = prices[-1]
    price_change = predicted_price - current_price
    percent_change = (price_change / current_price) * 100
    is_up = predicted_price > current_price
    
    # Format hasil prediksi
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #1e40af; margin-top: 0;">🔮 Hasil Prediksi</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div>
            <div style="font-size: 0.9rem; color: #4b5563;">Harga Terakhir</div>
            <div class="result-value">${current_price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div>
            <div style="font-size: 0.9rem; color: #4b5563;">Prediksi Besok</div>
            <div class="result-value">${predicted_price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        change_color = "#10b981" if is_up else "#ef4444"
        change_icon = "▲" if is_up else "▼"
        st.markdown(f"""
        <div>
            <div style="font-size: 0.9rem; color: #4b5563;">Perubahan</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {change_color};">
                {change_icon} ${abs(price_change):,.2f} ({abs(percent_change):.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top: 1.5rem; text-align: center;">
    """, unsafe_allow_html=True)
    
    if is_up:
        st.markdown('<div class="trend-up"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline><polyline points="16 7 22 7 22 13"></polyline></svg> TREN NAIK</div>', unsafe_allow_html=True)
        st.markdown('<p style="color: #065f46;">Berdasarkan prediksi, harga diperkirakan akan mengalami kenaikan.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="trend-down"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"></polyline><polyline points="16 17 22 17 22 11"></polyline></svg> TREN TURUN</div>', unsafe_allow_html=True)
        st.markdown('<p style="color: #991b1b;">Berdasarkan prediksi, harga diperkirakan akan mengalami penurunan.</p>', unsafe_allow_html=True)
    
    st.markdown("""
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div style="margin-top: 2rem; padding: 1rem; background-color: #fffbeb; border-radius: 8px; border-left: 4px solid #f59e0b;">
        <p style="color: #92400e; margin: 0;"><strong>⚠️ Disclaimer:</strong> Prediksi ini didasarkan pada model AI dan data historis. Hasil tidak menjamin akurasi 100%. Selalu lakukan penelitian tambahan sebelum membuat keputusan investasi.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("Silakan lengkapi semua harga 10 hari terakhir untuk mendapatkan prediksi.")

st.markdown('</div>', unsafe_allow_html=True)