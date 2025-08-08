import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
import pandas as pd

# Streamlit page configuration
st.set_page_config(page_title="Şebeke Analizi ve Kompanzasyon Çözümleri", layout="wide")

# PDF için Türkçe karakter desteği: Windows'ta Arial, yoksa DejaVu Sans; başarısızsa Helvetica
FONT_REG = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
try:
    pdfmetrics.registerFont(TTFont("Arial", r"C:\\Windows\\Fonts\\arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\\Windows\\Fonts\\arialbd.ttf"))
    FONT_REG, FONT_BOLD = "Arial", "Arial-Bold"
except Exception:
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", "DejaVuSans-Bold.ttf"))
        FONT_REG, FONT_BOLD = "DejaVuSans", "DejaVuSans-Bold"
    except Exception:
        pass

# Başlık ve bilgi
st.title("Şebeke Analizi ve Kompanzasyon Çözümleri")
st.info("Bu analiz, IEC 61000-4-30 Class A standartlarına uygun cihazlarla yapılan ölçümlerle gerçekleştirilmelidir.")

# IEC Harmonik Limitleri
HARMONIC_LIMITS = {
    3: 5.0, 5: 6.0, 7: 5.0, 9: 1.5, 11: 3.5, 13: 3.0,
    15: 0.5, 17: 2.0, 19: 1.5, 21: 0.5, 23: 1.5, 25: 1.5
}

# Standart kondansatör adım büyüklükleri (kVAr)
STANDARD_STEP_SIZES = [2.5, 5.0, 10.0, 12.5, 25.0, 50.0]

def pick_nearest_step_size(target_kvar: float) -> float:
    if target_kvar <= 0:
        return STANDARD_STEP_SIZES[0]
    return min(STANDARD_STEP_SIZES, key=lambda s: abs(s - target_kvar))

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# Dinamik puanlama sistemi
def calculate_solution_score(data, cosphi, target_pf, thdi, harmonics, calculated_i_neutral, I_avg_sys, apparent_power, transformer_rating):
    score = 0
    
    # PF boşluğu (0-100)
    pf_gap = (target_pf - cosphi) * 100
    score += pf_gap * 0.25
    
    # THDi ve harmonik cezaları
    thdi_penalty = thdi * 0.3
    score += thdi_penalty * 0.2
    
    harmonic_penalties = {3: 1.4, 5: 1.2, 7: 1.0, 11: 0.8, 13: 0.8}
    for h in harmonics:
        if h['order'] in harmonic_penalties:
            limit = HARMONIC_LIMITS.get(h['order'], 0.5)
            if h['percentage'] > limit:
                score += h['percentage'] * harmonic_penalties[h['order']] * 0.2
    
    # Nötr akımı oranı
    if I_avg_sys > 0:
        neutral_ratio = (calculated_i_neutral / I_avg_sys) * 100
        score += neutral_ratio * 0.15
    
    # Trafo yüklenmesi
    if transformer_rating > 0:
        transformer_load = (apparent_power / transformer_rating) * 100
        if transformer_load > 80:
            score += (transformer_load - 80) * 0.1
    
    return min(score, 100)

# Nötr akımı hesabı düzeltmesi
def calculate_neutral_current_corrected(harmonics, I_total, thdi):
    # Temel akım bileşeni
    I1 = I_total / np.sqrt(1 + (thdi/100)**2)
    
    # Triplen harmonikler (3, 9, 15, 21...) nötr akımına katkı yapar
    triplen_harmonics = [h for h in harmonics if h['order'] % 3 == 0]
    
    if not triplen_harmonics:
        # Eğer triplen harmonik yoksa, THDi'den tahmin et
        I_neutral_estimated = I_total * (thdi/100) * 0.3  # Yaklaşık değer
        return I_neutral_estimated
    
    # Triplen harmoniklerin toplam katkısı
    I_neutral = 0
    for h in triplen_harmonics:
        I_h = (h['percentage']/100) * I1
        I_neutral += I_h
    
    # Fazlar arası dağılım katsayısı
    distribution_factor = 0.85
    I_neutral = I_neutral * distribution_factor
    
    return I_neutral

# AHF boyutlandırması
def calculate_ahf_rating(harmonics, I_nominal, ahf_percentage=30):
    # Harmonik RMS akımı
    harmonic_currents = []
    for h in harmonics:
        I_h = I_nominal * (h['percentage'] / 100)
        harmonic_currents.append(I_h)
    
    # Toplam harmonik akım
    I_harmonic_total = np.sqrt(sum(I**2 for I in harmonic_currents))
    
    # AHF akım kapasitesi
    ahf_current_capacity = I_nominal * (ahf_percentage / 100)
    
    return min(I_harmonic_total, ahf_current_capacity)

# Rezonans riski hesaplama
def calculate_resonance_risk(Q_c, S_sc, f_nominal=50):
    # Q_c sıfır ise rezonans riski yok
    if Q_c <= 0:
        return False, 0, None
    
    # Paralel rezonans frekansı
    f_resonance = f_nominal * np.sqrt(S_sc / Q_c)
    
    # Harmonik sıraları
    harmonic_orders = [3, 5, 7, 11, 13, 17, 19, 23, 25]
    harmonic_frequencies = [f_nominal * h for h in harmonic_orders]
    
    # Risk hesaplama
    risk_threshold = 0.1  # %10 tolerans
    resonance_risk = False
    closest_harmonic = None
    
    for i, f_harmonic in enumerate(harmonic_frequencies):
        if abs(f_resonance - f_harmonic) / f_harmonic < risk_threshold:
            resonance_risk = True
            closest_harmonic = harmonic_orders[i]
            break
    
    return resonance_risk, f_resonance, closest_harmonic

# Tekil limit uyumu raporu
def generate_harmonic_compliance_report(harmonics, filter_reduction):
    compliance_table = []
    
    for h in harmonics:
        original = h['percentage']
        reduced = original * filter_reduction.get(h['order'], 1.0)
        limit = HARMONIC_LIMITS.get(h['order'], 0.5)
        compliant = reduced <= limit
        
        compliance_table.append({
            'Harmonik': h['order'],
            'Önceki': f"{original:.1f}%",
            'Sonra': f"{reduced:.1f}%",
            'Limit': f"{limit:.1f}%",
            'Uygun': "✅" if compliant else "❌"
        })
    
    return compliance_table

# SVG vs AHF ayrımı
def determine_solution_type(pf_gap, thdi, load_dynamics="Orta"):
    svg_score = 0
    ahf_score = 0
    
    # PF boşluğu büyükse SVG tercih
    if pf_gap > 0.15:  # %15'den büyük boşluk
        svg_score += 30
    
    # Harmonikler yüksekse AHF tercih
    if thdi > 20:
        ahf_score += 40
    elif thdi > 10:
        ahf_score += 20
    
    # Yük dinamikliği
    dynamics_multiplier = {"Statik": 0.5, "Orta": 1.0, "Hızlı": 1.5}
    multiplier = dynamics_multiplier.get(load_dynamics, 1.0)
    svg_score *= multiplier
    
    if svg_score > ahf_score:
        return "SVG"
    elif ahf_score > svg_score:
        return "AHF"
    else:
        return "SVG + AHF"

# Girdi formu
with st.form("input_form"):
    phase_type = st.selectbox(
        "Şebeke Tipi", ["Tek Fazlı", "Üç Fazlı"],
        help="Şebekenin tek fazlı mı yoksa üç fazlı mı olduğunu seçin."
    )

    if phase_type == "Üç Fazlı":
        st.subheader("Faz Değerleri")
        col_v = st.columns(3)
        v_l1 = col_v[0].number_input(
            "L1 Gerilim (V)", min_value=0.0, 
            help="L1 fazının RMS gerilim değeri (VAC)."
        )
        v_l2 = col_v[1].number_input(
            "L2 Gerilim (V)", min_value=0.0, 
            help="L2 fazının RMS gerilim değeri (VAC)."
        )
        v_l3 = col_v[2].number_input(
            "L3 Gerilim (V)", min_value=0.0, 
            help="L3 fazının RMS gerilim değeri (VAC)."
        )

        col_i = st.columns(3)
        i_l1 = col_i[0].number_input(
            "L1 Akım (A)", min_value=0.0, 
            help="L1 fazının RMS akım değeri (A)."
        )
        i_l2 = col_i[1].number_input(
            "L2 Akım (A)", min_value=0.0, 
            help="L2 fazının RMS akım değeri (A)."
        )
        i_l3 = col_i[2].number_input(
            "L3 Akım (A)", min_value=0.0, 
            help="L3 fazının RMS akım değeri (A)."
        )

        i_neutral = st.number_input(
            "Nötr Akım (A)", min_value=0.0, 
            help="Nötr iletkenindeki RMS akım değeri (A)."
        )
    else:
        v = st.number_input(
            "Gerilim (V)", min_value=0.0, 
            help="Şebekenin RMS gerilim değeri (VAC)."
        )
        i = st.number_input(
            "Akım (A)", min_value=0.0, 
            help="Şebekenin RMS akım değeri (A)."
        )
        i_neutral = st.number_input(
            "Nötr Akım (A)", min_value=0.0, 
            help="Nötr iletkenindeki RMS akım değeri (A)."
        )

    st.subheader("Sistem Parametreleri")
    col_p = st.columns(2)
    with col_p[0]:
        total_power = st.number_input(
            "Toplam Kurulu Güç (kVA)", min_value=0.0, 
            help="Tesisin toplam kurulu gücü (kVA)."
        )
        active_power = st.number_input(
            "Aktif Güç (kW)", min_value=0.0, 
            help="Ölçülen aktif güç (kW)."
        )
        reactive_power = st.number_input(
            "Reaktif Güç (kVAR)", min_value=0.0, 
            help="Ölçülen reaktif güç (kVAR)."
        )
        reactive_type = st.selectbox(
            "Reaktif Güç Tipi", ["Endüktif", "Kapasitif"],
            help="Reaktif gücün endüktif mi kapasitif mi olduğunu seçin."
        )
    with col_p[1]:
        cosphi = st.number_input(
            "Cosφ (Güç Faktörü)", min_value=0.0, max_value=1.0, 
            help="Ölçülen güç faktörü (cosφ)."
        )
        target_pf = st.number_input(
            "Hedef Cosφ", min_value=0.0, max_value=1.0,
            help="Önerilen güç faktörü hedef değeri."
        )
        thdv = st.number_input(
            "THDv (%)", min_value=0.0, 
            help="Gerilim harmonik distorsiyon oranı (%)."
        )
        thdi = st.number_input(
            "THDi (%)", min_value=0.0, 
            help="Akım harmonik distorsiyon oranı (%)."
        )
        transformer_rating = st.number_input(
            "Trafo Gücü (kVA)", min_value=0.0, 
            help="Trafo nominal gücü (kVA)."
        )
                # — Yeni: Şebeke Şalteri ve N–T Voltajı —
        breaker_rating = st.number_input(
            "Şebeke Şalteri (A)", min_value=0,
            help="Ana şalterinizin akım sınırlaması (A)."
        )
        ng_voltage = st.number_input(
            "Nötr-Topraklama Voltajı (V)", min_value=0.0, format="%.1f",
            help="Tesisinizdeki nötr–toprak potansiyel farkı (V)."
        )


    st.subheader("🔌 Mevcut Kompanzasyon (Opsiyonel)")
    existing_q = st.number_input(
        "Toplam Kapasitif Reaktif Güç (kVAr)",
        min_value=0.0, step=1.0, format="%.1f",
        key="existing_q",
        help="Sisteme şu anda bağlı kondansatör bankalarının toplam reaktif gücü"
    )
    filter_type = st.selectbox(
        "Filtre Türü",
        ["Yok", "Pasif Filtre", "Aktif Filtre"],
        key="filter_type",
        help="Varsa sisteminizdeki filtre teknolojisi"
    )
    filter_q = st.number_input(
        "Filtre Reaktif Gücü (kVAr)",
        min_value=0.0, step=1.0, format="%.1f",
        key="filter_q",
        help="Filtre varsa, onun kapasitesi"
    )
    connection = st.selectbox(
        "Bağlantı Tipi",
        ["Yıldız (Wye)", "Üçgen (Delta)"],
        key="conn_type",
        help="Kondansatör bankalarının faz bağlantı şekli"
    )
    nominal_v = st.number_input(
        "Nominal Faz Gerilimi (V)",
        min_value=100.0, step=1.0, format="%.0f",
        key="nominal_v",
        help="Örneğin 400 V"
    )
    steps = st.number_input(
        "Adım / Banka Sayısı",
        min_value=1, step=1,
        key="bank_steps",
        help="Kaç adım şeklinde devreye alabiliyorsunuz?"
    )
    existing_pf = st.slider(
        "Mevcut Güç Faktörü (cos φ)",
        min_value=0.70, max_value=1.00, step=0.01,
        key="existing_pf",
        help="Ölçtüğünüz güncel güç faktörü"
    )
    load_dynamics = st.selectbox(
        "Yük Dinamikliği",
        ["Statik", "Orta", "Hızlı"],
        index=1,
        key="load_dynamics",
        help="Sistemdeki yük değişim hızı"
    )

    
    st.subheader("Harmonik Girişi")
    st.info("Harmonik yüzdeleri temel akım bileşenine göre verilmelidir.")
    harmonic_count = st.number_input(
        "Harmonik Sayısı", min_value=0, max_value=25, step=1,
        help="Girmek istediğiniz harmonik bileşen sayısı."
    )

    harmonics = []
    # Sadece odd harmonics (3,5,7,…25) listesi
    valid_orders = list(HARMONIC_LIMITS.keys())

    for k in range(harmonic_count):
        st.markdown(f"**{k+1}. Harmonik**")
        col1, col2 = st.columns(2)
        # 1) Harmonik sırası: selectbox ile “Seçiniz” opsiyonlu
        order_str = col1.selectbox(
            f"Harmonik Sırası #{k+1}",
            options=["Seçiniz"] + [str(o) for o in valid_orders],
            key=f"order_{k}",
            help="3, 5, 7, … gibi tek sayı harmonik sıralarından birini seçin."
        )
        # 2) Yüzde girişi: zorunlu, placeholder olmuyor ama boş bırakıp formu gönderemezsiniz
        percentage = col2.number_input(
            f"Harmonik Yüzdesi #{k+1} (%)",
            min_value=0.0, max_value=100.0, step=0.1,
            key=f"perc_{k}",
            help="Bu harmonik bileşenin temel akıma oranı (%)"
        )

        # Validasyon: kullanıcı seçmediyse hata mesajı gösterelim
        if order_str == "Seçiniz":
            st.error(f"{k+1}. harmonik sırasını seçmelisiniz.")
        else:
            harmonics.append({
                "order": int(order_str),
                "percentage": percentage
            })


    submitted = st.form_submit_button(
        "Analizi Başlat",
        help="Girdiğiniz değerlerle şebeke analizini ve öneri algoritmasını çalıştırır."
    )


if submitted:
    # Veri toplama
    if phase_type == "Üç Fazlı":
        voltages = [v_l1, v_l2, v_l3]
        currents = [i_l1, i_l2, i_l3]
        data = {
            "phase_type": phase_type,
            "voltages": voltages,
            "currents": currents,
            "i_neutral": i_neutral,
            "total_power": total_power,
            "active_power": active_power,
            "reactive_power": reactive_power,
            "reactive_type": reactive_type,
            "cosphi": cosphi,
            "target_pf": target_pf,
            "thdv": thdv,
            "thdi": thdi,
            "harmonics": harmonics,
            "transformer_rating": transformer_rating,
            "breaker_rating": breaker_rating,
            "ng_voltage": ng_voltage

        }
    else:
        data = {
            "phase_type": phase_type,
            "voltage": v,
            "current": i,
            "i_neutral": i_neutral,
            "total_power": total_power,
            "active_power": active_power,
            "reactive_power": reactive_power,
            "reactive_type": reactive_type,
            "cosphi": cosphi,
            "target_pf": target_pf,
            "thdv": thdv,
            "thdi": thdi,
            "harmonics": harmonics,
            "transformer_rating": transformer_rating,
            "breaker_rating": breaker_rating,
            "ng_voltage": ng_voltage

        }

    # Girdi doğrulama
    errors = []
    warnings = []
    
    # Pozitif değer kontrolü
    if phase_type == "Üç Fazlı":
        if any(vv <= 0 for vv in voltages) or any(ii <= 0 for ii in currents) or i_neutral < 0:
            errors.append("Tüm gerilim ve akım değerleri pozitif olmalıdır.")
    else:
        if v <= 0 or i <= 0 or i_neutral < 0:
            errors.append("Gerilim ve akım değerleri pozitif olmalıdır.")

    # Aralık kontrolü
    if (total_power <= 0 or active_power < 0 or reactive_power < 0 or
        cosphi <= 0 or cosphi > 1 or thdv < 0 or thdv > 100 or thdi < 0 or thdi > 100):
        errors.append("Sistem parametreleri geçerli aralıklarda olmalıdır.")

    # Matematiksel tutarlılık
    apparent_power = np.sqrt(active_power**2 + reactive_power**2)
    distortion_power = apparent_power * (thdi / 100)
    calculated_cosphi = active_power / apparent_power if apparent_power > 0 else 0
    
    # ±%3 tolerans
    if apparent_power > 0 and abs(calculated_cosphi - cosphi) > 0.03:
        errors.append(
            f"Girilen cosφ ({cosphi:.2f}) hesaplanan değerle ({calculated_cosphi:.2f}) uyuşmuyor."
        )

    # Üç faz dengesizlik kontrolü
    if phase_type == "Üç Fazlı":
        V_avg = np.mean(voltages)
        if V_avg > 0:
            V_dev = [abs(vv - V_avg) / V_avg for vv in voltages]
            unbalance_v = max(V_dev) * 100
            if unbalance_v > 5:
                warnings.append(f"Faz gerilimleri %{unbalance_v:.1f} dengesiz (limit %5).")
        
        I_avg = np.mean(currents)
        if I_avg > 0:
            I_dev = [abs(ii - I_avg) / I_avg for ii in currents]
            unbalance_i = max(I_dev) * 100
            if unbalance_i > 5:
                warnings.append(f"Faz akımları %{unbalance_i:.1f} dengesiz (limit %5).")

    # Harmonik tutarlılık (±2% tolerans)
    calculated_thdi = np.sqrt(sum((h['percentage'])**2 for h in harmonics))
    if abs(calculated_thdi - thdi) > 2.0:
        warnings.append(
            f"Harmonikler THDi ile uyuşmuyor: Hesaplanan {calculated_thdi:.2f}%, Girilen {thdi:.2f}%"
        )
    else:
        thdi = calculated_thdi  # Hesaplanan değeri kullan

    # Harmonik limit kontrolü
    harmonic_violations = []
    for h in harmonics:
        limit = HARMONIC_LIMITS.get(h['order'], 0.5)  # Bilinmeyen harmonikler için %0.5 limit
        if h['percentage'] > limit:
            harmonic_violations.append(f"{h['order']}. harmonik: %{h['percentage']} > %{limit} (IEC 61000-3-6)")

    # Trafo yüklenmesi ve diğer sistem kontrolleri
    if phase_type == "Üç Fazlı":
        V_avg_sys = float(np.mean(voltages)) if 'voltages' in locals() else 0.0
        I_avg_sys = float(np.mean(currents)) if 'currents' in locals() else 0.0
        S_est_kVA = np.sqrt(3) * V_avg_sys * I_avg_sys / 1000 if V_avg_sys > 0 and I_avg_sys > 0 else 0.0
    else:
        V_avg_sys = float(v)
        I_avg_sys = float(i)
        S_est_kVA = V_avg_sys * I_avg_sys / 1000 if V_avg_sys > 0 and I_avg_sys > 0 else 0.0

    if total_power > 0 and apparent_power > 0 and active_power > apparent_power:
        warnings.append("Aktif güç, görünür güçten büyük görünüyor. Ölçüm veya giriş değerlerini kontrol edin.")

    if transformer_rating > 0:
        transformer_load_pct = 100.0 * (apparent_power / transformer_rating)
        if transformer_load_pct > 100:
            warnings.append(f"Trafo aşırı yük: %{transformer_load_pct:.0f}")
        elif transformer_load_pct > 80:
            warnings.append(f"Trafo yüklenmesi yüksek: %{transformer_load_pct:.0f}")
    else:
        transformer_load_pct = 0.0

    if S_est_kVA > 0 and apparent_power > 0:
        diff_pct = abs(S_est_kVA - apparent_power) / apparent_power * 100
        if diff_pct > 20:
            warnings.append("Güç tutarsızlığı: Ölçülen S ile V·I tahmini arasında >%20 sapma var.")

    if breaker_rating > 0 and I_avg_sys > breaker_rating:
        warnings.append("Şebeke şalteri akım sınırı aşılıyor.")

    if ng_voltage >= 5.0:
        warnings.append("Nötr-Toprak voltajı yüksek (≥5 V). Topraklama/Nötr bağlantılarını kontrol edin.")
    elif ng_voltage >= 3.0:
        warnings.append("Nötr-Toprak voltajı yükselmiş (≥3 V).")

    # Nötr akım analizi (düzeltilmiş hesaplama)
    if phase_type == "Üç Fazlı":
        I_avg_sys = np.mean(currents)
        calculated_i_neutral = calculate_neutral_current_corrected(harmonics, I_avg_sys, thdi)
        if abs(calculated_i_neutral - i_neutral) > 5:
            warnings.append(f"Hesaplanan nötr akımı {calculated_i_neutral:.1f}A, ölçülen {i_neutral}A")
    else:
        calculated_i_neutral = 0
        I_avg_sys = i

    # Rezonans riski analizi (geliştirilmiş)
    # Kullanıcıdan rezonans parametreleri alınabilir
    Q_c_total = st.session_state.get("existing_q", 0.0) + st.session_state.get("filter_q", 0.0)
    S_sc = 100.0  # Varsayılan kısa devre gücü (MVA)
    
    resonance_risk, f_resonance, closest_harmonic = calculate_resonance_risk(Q_c_total, S_sc)
    
    if resonance_risk:
        warnings.append(f"Rezonans riski: {f_resonance:.0f} Hz ≈ {closest_harmonic}. harmonik")

    # Hata ve uyarıları göster
    if errors:
        for err in errors:
            st.error(err)
        st.stop()
    
    if warnings:
        for warn in warnings:
            st.warning(warn)
    
    if harmonic_violations:
        st.subheader("Harmonik Limit İhlalleri")
        for violation in harmonic_violations:
            st.error(violation)

    st.success("Veriler doğrulandı, analiz başlatılıyor...")

    # Analiz
    # PF hedefi daha düşükse kompanzasyon gerekmez
    pf_issue = cosphi < target_pf
    harmonic_issue = thdi > 5
    resonance_issue = resonance_risk

    # Dinamik puanlama hesaplama
    system_score = calculate_solution_score(data, cosphi, target_pf, thdi, harmonics, calculated_i_neutral, I_avg_sys, apparent_power, transformer_rating)
    
    # Çözüm önerileri (dinamik puanlama ile)
    solutions = []
    
    # SVG vs AHF ayrımı
    pf_gap = target_pf - cosphi
    load_dynamics_value = st.session_state.get("load_dynamics", "Orta")
    solution_type = determine_solution_type(pf_gap, thdi, load_dynamics_value)
    
    if pf_issue:
        if solution_type in ["SVG", "SVG + AHF"]:
            solutions.append({
                "name": "SVG (Static Var Generator)",
                "reason": "Dinamik reaktif güç kontrolü ve hızlı tepki",
                "suitability": 4,
                "score": system_score * 0.8
            })
        else:
            solutions.append({
                "name": "Klasik Kompanzasyon",
                "reason": "Güç faktörü düzeltmesi için kondansatör bankı",
                "suitability": 3,
                "score": system_score * 0.6
            })
    
    if harmonic_issue:
        if solution_type in ["AHF", "SVG + AHF"]:
            # AHF boyutlandırması
            ahf_rating = calculate_ahf_rating(harmonics, I_avg_sys, 30)
            solutions.append({
                "name": f"Aktif Filtre ({ahf_rating:.1f}A)",
                "reason": f"Tüm harmonik sıraları için etkili çözüm - {ahf_rating:.1f}A kapasite",
                "suitability": 5,
                "score": system_score * 0.9
            })
        else:
            solutions.append({
                "name": "Pasif Filtre",
                "reason": "Belirli harmonik sıraları için ekonomik çözüm",
                "suitability": 3,
                "score": system_score * 0.5
            })
    
    if resonance_issue:
        solutions.append({
            "name": "Detune Reaktör",
            "reason": "Rezonans riskini azaltmak için",
            "suitability": 4,
            "score": system_score * 0.7
        })
    
    # Tekrarlanan çözümleri kaldır ve puanlama ile sırala
    unique = {}
    for sol in solutions:
        if sol["name"] not in unique or sol["score"] > unique[sol["name"]]["score"]:
            unique[sol["name"]] = sol
    unique_solutions = sorted(unique.values(), key=lambda x: x["score"], reverse=True)

    # Reaktif güç hesaplamaları
    is_inductive = reactive_type == "Endüktif"
    phi1 = np.arccos(clamp(cosphi, 1e-6, 1)) if cosphi > 0 else 0.0
    phi2 = np.arccos(clamp(target_pf, 1e-6, 1)) if target_pf > 0 else 0.0
    Q_target = active_power * np.tan(phi2)
    # Endüktif yük için gereken kapasitif kVAr: P*(tan phi1 - tan phi2)
    if target_pf <= cosphi:
        Q_required_theoretical = 0.0
        pf_issue = False
    else:
        Q_required_theoretical = max(0.0, active_power * (np.tan(phi1) - np.tan(phi2)))

    # Mevcut ekipmanın etkisi
    existing = st.session_state.existing_q if "existing_q" in st.session_state else 0.0
    filter_q_val = st.session_state.filter_q if "filter_q" in st.session_state else 0.0
    Q_comp = max(0.0, Q_required_theoretical - existing - filter_q_val)

    # Adım / konfigürasyon önerisi
    steps_count = max(1, int(st.session_state.get("bank_steps", 3)))
    step_size_guess = Q_comp / steps_count if steps_count > 0 else Q_comp
    step_size = pick_nearest_step_size(step_size_guess)
    config_text = f"{steps_count} x {step_size:.1f} kVAr"

    
    # Harmonik filtrasyon etkisi (dinamik)
    order_reduction = {}
    S_nonzero = apparent_power if apparent_power > 0 else 1.0
    # Simülasyon için, kullanıcı filtre seçmemişse fakat harmonik sorunu varsa
    # önerilen çözüm varsayımıyla (Aktif Filtre) etkiyi göster.
    sim_filter_type = st.session_state.get("filter_type", "Yok")
    sim_filter_q = st.session_state.get("filter_q", 0.0)
    if sim_filter_type == "Yok" and harmonic_issue:
        sim_filter_type = "Aktif Filtre"
        # kaba boyutlandırma: S'nin %20-40'ı aralığında hedefle
        sim_filter_q = max(sim_filter_q, 0.3 * S_nonzero)

    sizing_ratio = clamp((sim_filter_q) / S_nonzero, 0.0, 1.0)
    if sim_filter_type == "Yok":
        for h in HARMONIC_LIMITS:
            order_reduction[h] = 1.0
    elif sim_filter_type == "Pasif Filtre":
        base = 1.0 - 0.6 * sizing_ratio
        for h in HARMONIC_LIMITS:
            if h in [3, 5, 7]:
                order_reduction[h] = clamp(base, 0.3, 1.0)
            elif h in [11, 13]:
                order_reduction[h] = clamp(base + 0.2, 0.5, 1.0)
            else:
                order_reduction[h] = clamp(0.9, 0.5, 1.0)
    else:  # Aktif Filtre
        # Daha agresif model: oran 0.3 için ~%60 düşüş, 0.5 için alt sınır
        base = 1.0 - 2.0 * sizing_ratio
        for h in HARMONIC_LIMITS:
            order_reduction[h] = clamp(base, 0.2, 1.0)
    
    # Simülasyon
    f = 50  # Hz
    t = np.linspace(0, 0.04, 1000)  # 2 periyot
    # RMS hesaplama
    if phase_type == "Üç Fazlı":
        I_total_rms = np.sqrt((i_l1**2 + i_l2**2 + i_l3**2) / 3)
    else:
        I_total_rms = i
        
    I1_rms = I_total_rms / np.sqrt(1 + (thdi/100)**2)
    sqrt2 = np.sqrt(2)

    # Temel akım bileşeni
    phase_angle = np.arccos(cosphi) if is_inductive else -np.arccos(cosphi)
    i_fund = sqrt2 * I1_rms * np.sin(2 * np.pi * f * t + phase_angle)
    
    # Harmonik bileşenleri
    i_harm = np.zeros_like(t)
    harmonic_components = []
    for h in harmonics:
        I_n_rms = (h['percentage'] / 100) * I1_rms
        reduction = order_reduction.get(h['order'], 1.0)
        harmonic_components.append({
            "order": h['order'],
            "original": I_n_rms,
            "reduced": I_n_rms * reduction
        })
        i_harm += sqrt2 * I_n_rms * np.sin(2 * np.pi * f * h['order'] * t)
    
    i_total = i_fund + i_harm

    # Kompanzasyon sonrası
    i_harm_comp = np.zeros_like(t)
    for h in harmonics:
        I_n_rms = (h['percentage'] / 100) * I1_rms
        reduction = order_reduction.get(h['order'], 1.0)
        i_harm_comp += sqrt2 * I_n_rms * reduction * np.sin(2 * np.pi * f * h['order'] * t)
    
    # Faz kayması etkisi
    new_phase = np.arccos(target_pf) if is_inductive else -np.arccos(target_pf)
    i_fund_comp = sqrt2 * I1_rms * np.sin(2 * np.pi * f * t + new_phase)
    i_total_comp = i_fund_comp + i_harm_comp

    # Harmonik spektrumu
    harmonic_orders = [h['order'] for h in harmonics]
    harmonic_percentages = [h['percentage'] for h in harmonics]
    reduced_percentages = [h['percentage'] * order_reduction.get(h['order'], 1.0) for h in harmonics]

    # THDi sonrası tahmin
    thdi_after = np.sqrt(sum((p * order_reduction.get(o, 1.0))**2 for o, p in zip(harmonic_orders, harmonic_percentages)))

    # Nötr akımı değerlendirmesi (yüzde)
    if phase_type == "Üç Fazlı" and I_avg_sys > 0:
        neutral_pct = (calculated_i_neutral / I_avg_sys) * 100
    else:
        neutral_pct = 0.0

    # Görselleştirme
    # Özet blokları
    st.subheader("Reaktif Güç İhtiyacı")
    st.markdown(f"- Gerekli kompanzasyon: ~**{Q_comp:.1f} kVAr**")
    st.markdown(f"- Konfigürasyon: **{config_text}**")
    st.markdown(f"- Hedef PF: **{cosphi:.2f} → {target_pf:.2f}**")

    st.subheader("Harmonik Analiz")
    if harmonic_violations:
        st.markdown("- IEC limitleri: **İhlal var**")
    else:
        st.markdown("- IEC limitleri: **Uygun**")
    st.markdown(f"- THDi: **{thdi:.1f}% → {thdi_after:.1f}% (tahmini)**")
    if phase_type == "Üç Fazlı":
        st.markdown(f"- Nötr akımı: **{calculated_i_neutral:.1f} A ({neutral_pct:.1f}%)**")

    st.subheader("Trafo Yüklenmesi")
    st.markdown(f"- Mevcut yük: **{active_power:.0f} kW / {transformer_rating:.0f} kVA = %{(apparent_power/transformer_rating*100) if transformer_rating>0 else 0:.0f}**")
    st.markdown(f"- Harmonik kayıplar (tahmini): **{'<%3 (önemsiz)' if thdi_after < 3 else 'artabilir'}**")

    st.subheader("Koruma ve Güvenlik")
    st.markdown(f"- Şebeke şalteri: **{I_avg_sys:.0f} A / {breaker_rating:.0f} A**")
    st.markdown(f"- N–T voltajı: **{ng_voltage:.1f} V**")

    # Sistem skoru ve önerilen çözümler
    st.subheader("Sistem Analiz Skoru")
    st.metric("Dinamik Puan", f"{system_score:.1f}/100")
    
    st.subheader("Önerilen Çözümler")
    df_solutions = pd.DataFrame(unique_solutions)
    df_solutions["Uygunluk"] = df_solutions["suitability"].apply(lambda x: "★" * x)
    df_solutions["Skor"] = df_solutions["score"].apply(lambda x: f"{x:.1f}")
    st.dataframe(df_solutions[["name", "reason", "Uygunluk", "Skor"]], hide_index=True)

    # Simülasyon grafikleri
    st.subheader("Akım Dalga Formu Simülasyonu")
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(t, i_total, label='Toplam Akım')
    ax1.plot(t, i_fund, 'r--', label='Temel Bileşen')
    ax1.set_title("Kompanzasyon Öncesi Akım")
    ax1.set_xlabel("Zaman (s)")
    ax1.set_ylabel("Akım (A)")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t, i_total_comp, label='Toplam Akım')
    ax2.plot(t, i_fund_comp, 'r--', label='Temel Bileşen')
    ax2.set_title("Kompanzasyon Sonrası Akım")
    ax2.set_xlabel("Zaman (s)")
    ax2.set_ylabel("Akım (A)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig1)

    # Harmonik spektrumu
    st.subheader("Harmonik Spektrumu")
    fig2, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = np.arange(len(harmonic_orders))
    ax.bar(x - width/2, harmonic_percentages, width, label='Ölçülen')
    ax.bar(x + width/2, reduced_percentages, width, label='Filtre Sonrası')
    
    # Limit çizgileri
    for i, order in enumerate(harmonic_orders):
        limit = HARMONIC_LIMITS.get(order, 0.5)
        ax.axhline(y=limit, color='r', linestyle='--', alpha=0.5)
        ax.text(x[i] - 0.5, limit + 0.5, f'Limit: {limit}%', color='r', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(harmonic_orders)
    ax.set_xlabel("Harmonik Sırası")
    ax.set_ylabel("Yüzde (%)")
    ax.set_title("Harmonik Dağılımı ve Filtre Etkisi")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig2)

    # Tekil limit uyumu raporu
    st.subheader("Harmonik Limit Uyumu Raporu")
    compliance_report = generate_harmonic_compliance_report(harmonics, order_reduction)
    df_compliance = pd.DataFrame(compliance_report)
    st.dataframe(df_compliance, hide_index=True)

    # Fazör diyagramı
    st.subheader("Güç Faktörü Düzeltme")
    fig3, ax = plt.subplots(figsize=(8, 8))
    
    # Önceki durum
    S = apparent_power
    P = active_power
    Q = reactive_power if is_inductive else -reactive_power
    
    ax.quiver(0, 0, P, Q, angles='xy', scale_units='xy', scale=1, color='b', label='Önceki')
    ax.text(P*1.05, Q*1.05, f'cosφ={cosphi:.2f}', color='b')
    
    # Sonraki durum
    Q_target_adj = Q_target if is_inductive else -Q_target
    ax.quiver(0, 0, P, Q_target_adj, angles='xy', scale_units='xy', scale=1, color='r', label='Sonra')
    ax.text(P*1.05, Q_target_adj*1.05, f'cosφ={target_pf:.2f}', color='r')
    
    # Eksenler
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    ax.set_xlim(0, S*1.2)
    ax.set_ylim(min(Q, Q_target_adj)*1.2, max(Q, Q_target_adj)*1.2)
    ax.set_xlabel("Aktif Güç (kW)")
    ax.set_ylabel("Reaktif Güç (kVAr)")
    ax.set_title("Güç Fazör Diyagramı")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig3)

    # Sonuç özeti
    st.subheader("Sonuç Özeti")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cosφ", f"{cosphi:.2f} → {target_pf:.2f}")
    with col2:
        st.metric("Reaktif Güç (kVAR)", f"{reactive_power:.1f} → {Q_target:.1f}")
    with col3:
        st.metric("THDi (%)", f"{thdi:.1f} → {thdi_after:.1f}")
    with col4:
        st.metric("Gerekli Kompanzasyon", f"{Q_comp:.1f} kVAR")
        # — Yeni: Ek Sistem Parametreleri —
    st.subheader("Ek Sistem Parametreleri")
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Şebeke Şalteri (A)", f"{data['breaker_rating']}")
    with col6:
        st.metric("Nötr-Topraklama Voltajı (V)", f"{data['ng_voltage']:.1f}")
    

    # — Mevcut Kompanzasyon Özeti —
    if any([
        st.session_state.get("existing_q", 0) > 0,
        st.session_state.get("filter_type", "Yok") != "Yok",
        st.session_state.get("filter_q", 0) > 0
    ]):
        st.markdown("**Mevcut Kompanzasyon Bilgileri:**")
        df_comp = pd.DataFrame({
            "Parametre": [
                "Toplam Kap. Q (kVAr)",
                "Filtre Türü", "Filtre Q (kVAr)",
                "Bağlantı Tipi", "Nom. Gerilim (V)",
                "Adım Sayısı", "Mevcut cos φ"
            ],
            "Değer": [
                f"{st.session_state.get('existing_q', 0):.1f}",
                st.session_state.get("filter_type", "Yok"),
                f"{st.session_state.get('filter_q', 0):.1f}",
                st.session_state.get("conn_type", ""),
                f"{st.session_state.get('nominal_v', 0):.0f}",
                f"{st.session_state.get('bank_steps', 0)}",
                f"{st.session_state.get('existing_pf', 0):.2f}"
            ]
        })
        st.table(df_comp)
        st.info("Girdiğiniz mevcut kompanzasyon değerleri net Q ihtiyacına yansıtıldı.")


    # Risk analizi
    if resonance_risk:
        st.warning("**Rezonans Uyarısı:** Sistemde 4-13. harmoniklerde rezonans riski tespit edildi!")
    if harmonic_violations:
        st.error("**Harmonik Limit İhlali:** IEC 61000-3-6 standart limitleri aşılıyor!")

    # PDF raporu oluşturma
    def create_pdf_report(data, figures):
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        left_margin = 50
        right_margin = 50
        max_img_width = width - left_margin - right_margin

        def ensure_space(y_pos: float, block_height: float) -> float:
            if y_pos - block_height < 80:  # alt marj
                p.showPage()
                p.setFont(FONT_BOLD, 16)
                p.drawString(left_margin, height-50, "Şebeke Analizi ve Kompanzasyon Raporu")
                p.setFont(FONT_REG, 12)
                p.drawString(left_margin, height-80, f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
                p.line(left_margin, height-90, width-right_margin, height-90)
                return height - 120
            return y_pos

        def draw_figure(fig, y_pos: float, target_h: float = 280) -> float:
            nonlocal p
            y_pos = ensure_space(y_pos, target_h)
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            p.drawImage(ImageReader(img_buffer), left_margin, y_pos - target_h, width=max_img_width, height=target_h)
            return y_pos - (target_h + 20)
        
        # Başlık
        p.setFont(FONT_BOLD, 16)
        p.drawString(left_margin, height-50, "Şebeke Analizi ve Kompanzasyon Raporu")
        p.setFont(FONT_REG, 12)
        p.drawString(left_margin, height-80, f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        p.line(left_margin, height-90, width-right_margin, height-90)
        
        # Girdi verileri
        y = height-120
        p.setFont(FONT_BOLD, 14)
        p.drawString(left_margin, y, "Girdi Verileri")
        y -= 30
        
        p.setFont(FONT_REG, 10)
        if data["phase_type"] == "Üç Fazlı":
            p.drawString(left_margin, y, f"Şebeke Tipi: Üç Fazlı")
            y -= 20
            p.drawString(left_margin, y, f"Gerilimler: L1={data['voltages'][0]}V, L2={data['voltages'][1]}V, L3={data['voltages'][2]}V")
            y -= 20
            p.drawString(left_margin, y, f"Akımlar: L1={data['currents'][0]}A, L2={data['currents'][1]}A, L3={data['currents'][2]}A")
            y -= 20
            p.drawString(left_margin, y, f"Nötr Akımı: {data['i_neutral']}A")
            y -= 20
        else:
            p.drawString(left_margin, y, f"Şebeke Tipi: Tek Fazlı, Gerilim={data['voltage']}V, Akım={data['current']}A")
            y -= 20
            p.drawString(left_margin, y, f"Nötr Akımı: {data['i_neutral']}A")
            y -= 20
            
        p.drawString(left_margin, y, f"Toplam Kurulu Güç: {data['total_power']} kVA")
        y -= 20
        p.drawString(left_margin, y, f"Aktif Güç: {data['active_power']} kW")
        y -= 20
        p.drawString(left_margin, y, f"Reaktif Güç: {data['reactive_power']} kVAr ({data['reactive_type']})")
        y -= 20
        p.drawString(left_margin, y, f"Cosφ: {data['cosphi']:.2f} → Hedef: {data['target_pf']:.2f}")
        y -= 20
        p.drawString(left_margin, y, f"THDv: {data['thdv']:.1f}%")
        y -= 20
        p.drawString(left_margin, y, f"THDi: {data['thdi']:.1f}%")
        y -= 20
        p.drawString(left_margin, y, f"Trafo Gücü: {data['transformer_rating']} kVA")
        y -= 30
        y -= 20
        p.drawString(left_margin, y, f"Şebeke Şalteri: {data['breaker_rating']} A")
        y -= 20
        p.drawString(left_margin, y, f"Nötr-Topraklama Voltajı: {data['ng_voltage']} V")
        y -= 20
        p.drawString(left_margin, y, f"Yük Dinamikliği: {load_dynamics_value}")
        y -= 20


        
        # Özet blokları (Beklenen Çıktılar)
        p.setFont(FONT_BOLD, 14)
        p.drawString(left_margin, y, "Beklenen Çıktılar")
        y -= 25
        p.setFont(FONT_BOLD, 12)
        p.drawString(left_margin, y, "1) Reaktif Güç İhtiyacı")
        y -= 18
        p.setFont(FONT_REG, 10)
        p.drawString(left_margin+20, y, f"Gerekli kompanzasyon: ~{Q_comp:.1f} kVAr")
        y -= 14
        p.drawString(left_margin+20, y, f"Konfigürasyon: {config_text}")
        y -= 18
        p.setFont(FONT_BOLD, 12)
        p.drawString(left_margin, y, "2) Harmonik Analiz")
        y -= 18
        p.setFont(FONT_REG, 10)
        p.drawString(left_margin+20, y, f"IEC limitleri: {'İhlal var' if harmonic_violations else 'Uygun'}")
        y -= 14
        p.drawString(left_margin+20, y, f"THDi: {thdi:.1f}% → {thdi_after:.1f}% (tahmini)")
        y -= 14
        if data["phase_type"] == "Üç Fazlı":
            p.drawString(left_margin+20, y, f"Nötr akımı: {calculated_i_neutral:.1f} A ({neutral_pct:.1f}%)")
            y -= 14
        p.setFont(FONT_BOLD, 12)
        p.drawString(left_margin, y, "3) Trafo Yüklenmesi")
        y -= 18
        p.setFont(FONT_REG, 10)
        load_pct = (apparent_power/transformer_rating*100) if transformer_rating>0 else 0.0
        p.drawString(left_margin+20, y, f"Mevcut yük: {active_power:.0f} kW / {transformer_rating:.0f} kVA = %{load_pct:.0f}")
        y -= 14
        p.drawString(left_margin+20, y, f"Harmonik kayıplar: {'<%3 (önemsiz)' if thdi_after < 3 else 'artabilir'}")
        y -= 14
        p.drawString(left_margin+20, y, f"Sistem Analiz Skoru: {system_score:.1f}/100")
        y -= 24

        # Grafikler
        p.setFont(FONT_BOLD, 14)
        p.drawString(left_margin, y, "Analiz Sonuçları")
        y -= 30
        # Grafikleri PDF'ye ekle (sayfa taşmalarını yönet)
        for idx, fig in enumerate(figures):
            y = draw_figure(fig, y, target_h=300 if idx < 2 else 360)
        
        # Sonuç özeti
        p.showPage()
        p.setFont(FONT_BOLD, 16)
        p.drawString(left_margin, height-50, "Sonuçlar ve Öneriler")
        
        p.setFont(FONT_REG, 12)
        y = height-100
        p.drawString(left_margin, y, f"Güç Faktörü İyileştirme: {cosphi:.2f} → {target_pf:.2f}")
        y -= 30
        p.drawString(left_margin, y, f"THDi Azaltma: {thdi:.1f}% → {thdi_after:.1f}%")
        y -= 30
        p.drawString(left_margin, y, f"Gerekli Kompanzasyon Gücü: {Q_comp:.1f} kVAr")
        y -= 30
        
        if resonance_risk:
            p.setFillColorRGB(1, 0, 0)  # Kırmızı
            p.drawString(left_margin, y, "REZONANS RİSKİ: Sistemde rezonans riski tespit edildi!")
            p.setFillColorRGB(0, 0, 0)  # Siyah
            y -= 30
        
        if harmonic_violations:
            p.setFillColorRGB(1, 0, 0)  # Kırmızı
            p.drawString(left_margin, y, "HARMONİK LİMİT İHLALİ: IEC standartları aşılıyor!")
            p.setFillColorRGB(0, 0, 0)  # Siyah
            y -= 30
        
        # Önerilen çözümler
        y -= 30
        p.setFont(FONT_BOLD, 14)
        p.drawString(left_margin, y, "Önerilen Çözümler")
        y -= 30
        
        p.setFont(FONT_REG, 10)
        for i, sol in enumerate(unique_solutions):
            p.drawString(left_margin+20, y, f"{i+1}. {sol['name']} - Uygunluk: {'★' * sol['suitability']}")
            y -= 20
            p.drawString(left_margin+40, y, f"Neden: {sol['reason']}")
            y -= 30
            if y < 100:
                p.showPage()
                y = height - 50
        
        p.save()
        buffer.seek(0)
        return buffer

    # PDF oluştur ve indirme butonu
    with st.spinner("PDF rapor oluşturuluyor..."):
        pdf_buffer = create_pdf_report(data, [fig1, fig2, fig3])
        st.download_button(
            "📊 Raporu PDF Olarak İndir",
            data=pdf_buffer,
            file_name=f"sebeke_analiz_raporu_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )
