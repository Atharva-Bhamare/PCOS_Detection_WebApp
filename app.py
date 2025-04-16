import os
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import datetime
import io
import qrcode
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def generate_pdf_report(data: dict) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # === Clinic Logo ===
    import os
    logo_path = "clinic_logo.jpeg"
    if not os.path.exists(logo_path):
        logo_path = "clinic_logo.png"
    try:
        c.drawImage(logo_path, width - 2.5*inch, height - 1.2*inch, width=1.5*inch, height=1*inch, preserveAspectRatio=True)
    except:
        pass  # avoid crash if not found

    # === Watermark ===
    c.saveState()
    c.setFont("Helvetica", 40)
    c.setFillColorRGB(0.92, 0.92, 0.92)
    c.translate(width / 2, height / 2)
    c.rotate(45)
    c.drawCentredString(0, 0, "Wellness Forever")
    c.restoreState()

    # === Title ===
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(1 * inch, height - 1 * inch, "PCOS Prediction Report")

    # === Report Details ===
    c.setFont("Helvetica", 12)
    y = height - 1.5 * inch
    for key, value in data.items():
        c.drawString(1 * inch, y, f"{key}: {value}")
        y -= 0.4 * inch

    # === Doctor's Signature ===
    c.setFont("Helvetica", 11)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(1 * inch, 1.5 * inch, "Doctor's Signature: ______________________")

    # === QR Code (link to digital copy) ===
    import qrcode
    from PIL import Image
    qr_url = "https://yourclinicportal.com/report?id=" + data.get("Patient ID", "N/A")
    qr = qrcode.make(qr_url)
    qr_path = "temp_qr.png"
    qr.save(qr_path)

    try:
        c.drawImage(qr_path, width - 2*inch, 1*inch, width=1.2*inch, height=1.2*inch)
    except:
        pass

    # Finalize
    c.showPage()
    c.save()
    buffer.seek(0)

    # Clean up QR temp file
    if os.path.exists(qr_path):
        os.remove(qr_path)

    return buffer.read()

# ------------------------
# Load Model & Scaler
# ------------------------
with open("pcos_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ------------------------
# Session State for Logs
# ------------------------
if "log" not in st.session_state:
    st.session_state.log = []

# ------------------------
# UI Config
# ------------------------
st.set_page_config(page_title="PCOS Predictor", layout="centered")

st.markdown("""
<style>
.big-title { font-size: 2.5rem; font-weight: bold; text-align: center; color: #6C63FF; }
.subtext { font-size: 1.1rem; color: #555; text-align: center; }
.footer { font-size: 0.85rem; color: gray; text-align: center; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">PCOS Prediction Model</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">For Clinical Use Only - Doctor’s Interface</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------
# Patient Input Form
# ------------------------
st.markdown("### Patient Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", 10, 60, 25)
    bmi = st.number_input("BMI (kg/m²)", 10.0, 50.0, 22.0)
    irregular = st.radio("Menstrual Irregularity", ["No", "Yes"])
    menstrual_irregularity = 1 if irregular == "Yes" else 0

with col2:
    testosterone = st.number_input("Testosterone Level (ng/dL)", 10.0, 200.0, 60.0)
    afc = st.number_input("Antral Follicle Count", 0, 30, 12)

# ------------------------
# Predict Button
# ------------------------
if st.button("Predict PCOS"):
    features = np.array([[age, bmi, menstrual_irregularity, testosterone, afc]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    # ------------------------
    # Display Result
    # ------------------------
    st.markdown("### Prediction Result")
    if prediction == 1:
        st.error(f" **PCOS Likely Detected**\n\nProbability: `{prob:.2f}`")
        st.markdown("> _Recommend further clinical evaluation._")
        with st.expander("Health Tips"):
            st.markdown("""
            - Balanced, low-sugar diet
            - Regular physical activity
            - Medication if prescribed
            - Stress management
            """)
    else:
        st.success(f"**PCOS Not Likely**\n\nProbability: `{prob:.2f}`")
        st.markdown("> _Patient appears healthy._")

    # ------------------------
    # Log Session
    # ------------------------
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    entry = {
        "Time": timestamp,
        "Age": age,
        "BMI": bmi,
        "Menstrual": irregular,
        "Testosterone": testosterone,
        "AFC": afc,
        "Prediction": "PCOS" if prediction == 1 else "No PCOS",
        "Probability": f"{prob:.2f}"
    }
    st.session_state.log.append(entry)

    # ------------------------
    # Download Report Button
    # ------------------------
    pdf_data = {
        "Time": timestamp,
        "Age": age,
        "BMI": bmi,
        "Menstrual Irregularity": irregular,
        "Testosterone Level": testosterone,
        "Antral Follicle Count": afc,
        "Prediction": "PCOS Likely" if prediction == 1 else "No PCOS",
        "Probability": f"{prob:.2f}"
    }

    pdf_bytes = generate_pdf_report(pdf_data)

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"pcos_report_{timestamp.replace(':', '-')}.pdf",
        mime="application/pdf"
    )

    # ------------------------
    # Radar Chart with Comparison
    # ------------------------
    import plotly.graph_objects as go

    radar_labels = ['Age', 'BMI', 'Menstrual Irregularity', 'Testosterone', 'Follicle Count']

    # Patient values (scale menstrual irregularity to match others)
    patient_values = [age, bmi, menstrual_irregularity * 100, testosterone, afc]

    # Normal range reference values (approximate)
    normal_reference = [25, 22, 0, 50, 12]

    fig = go.Figure()

    # Add Patient data
    fig.add_trace(go.Scatterpolar(
        r=patient_values,
        theta=radar_labels,
        fill='toself',
        name='Patient',
        line=dict(color='mediumvioletred', width=3)
    ))

    # Add Normal Range data
    fig.add_trace(go.Scatterpolar(
        r=normal_reference,
        theta=radar_labels,
        fill='toself',
        name='Normal Range',
        line=dict(color='mediumseagreen', width=3, dash='dot')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, color="black", showline=True, showticklabels=True),
            angularaxis=dict(color="black")
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        showlegend=True,
        margin=dict(t=30, b=50, l=30, r=30)
    )

    st.markdown("### Patient vs Normal Profile")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Session History
# ------------------------
if st.session_state.log:
    st.markdown("---")
    st.markdown("### Session History")
    df_log = pd.DataFrame(st.session_state.log)
    st.dataframe(df_log, use_container_width=True)
    csv = df_log.to_csv(index=False).encode('utf-8')
    st.download_button("Download Session Log (CSV)", csv, "pcos_session_log.csv", "text/csv")
