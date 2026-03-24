import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="🏥 Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}
.main-header h1 {
    font-family: 'Nunito', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
}
.main-header p { color: #a8d8ea; font-size: 1.1rem; margin-top: 0.5rem; }

.sdg-badge {
    display: inline-block;
    background: linear-gradient(90deg, #4CAF50, #2E7D32);
    color: white;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-top: 0.8rem;
}
.result-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 2px solid #e94560;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 8px 32px rgba(233,69,96,0.2);
}
.result-disease {
    font-family: 'Nunito', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #e94560;
    margin: 0.5rem 0;
}
.result-confidence { font-size: 1.2rem; color: #a8d8ea; font-weight: 600; }

.info-card {
    background: linear-gradient(135deg, #0f3460, #16213e);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid #4CAF50;
    color: #ffffff;
}
.info-card h4 { color: #4CAF50; font-size: 1.1rem; margin-bottom: 0.5rem; }

.symptom-chip {
    display: inline-block;
    background: rgba(233,69,96,0.15);
    border: 1px solid #e94560;
    color: #e94560;
    padding: 0.3rem 0.8rem;
    border-radius: 30px;
    margin: 0.2rem;
    font-size: 0.85rem;
    font-weight: 600;
}
.warning-card {
    background: linear-gradient(135deg, #2d1b00, #3d2200);
    border: 1px solid #ff9800;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #ffcc80;
    font-size: 0.9rem;
    margin-top: 1rem;
}
.stButton > button {
    background: linear-gradient(135deg, #e94560, #c62a47) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.8rem 3rem !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(233,69,96,0.4) !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

.metric-box {
    background: linear-gradient(135deg, #0f3460, #1a1a2e);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid rgba(168,216,234,0.2);
}
.metric-box h2 {
    font-family: 'Nunito', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #e94560;
    margin: 0;
}
.metric-box p { color: #a8d8ea; font-size: 0.85rem; margin: 0.3rem 0 0 0; }
</style>
""", unsafe_allow_html=True)

disease_info = {
    "Flu": {"emoji": "🤧", "description_en": "Influenza is a contagious respiratory illness caused by influenza viruses.", "description_hi": "फ्लू एक संक्रामक श्वसन बीमारी है।", "treatment": "Rest, fluids, antiviral medications if needed.", "severity": "Moderate", "color": "#FF9800"},
    "Cold": {"emoji": "🤒", "description_en": "Common cold is a viral infection of the upper respiratory tract.", "description_hi": "सर्दी ऊपरी श्वसन पथ का एक वायरल संक्रमण है।", "treatment": "Rest, fluids, OTC cold medicines.", "severity": "Mild", "color": "#4CAF50"},
    "Dengue": {"emoji": "🦟", "description_en": "Dengue is a mosquito-borne viral disease causing high fever and severe pain.", "description_hi": "डेंगू एक मच्छर जनित वायरल बीमारी है।", "treatment": "Hospitalization may be needed, fluids, platelet monitoring.", "severity": "High", "color": "#F44336"},
    "Migraine": {"emoji": "🧠", "description_en": "Migraine is a neurological condition causing intense headaches.", "description_hi": "माइग्रेन एक न्यूरोलॉजिकल स्थिति है।", "treatment": "Pain relievers, triptans, rest in dark quiet room.", "severity": "Moderate", "color": "#9C27B0"},
    "Typhoid": {"emoji": "🌡️", "description_en": "Typhoid is a bacterial infection caused by Salmonella typhi.", "description_hi": "टाइफॉइड साल्मोनेला टाइफी बैक्टीरिया से होता है।", "treatment": "Antibiotics, hospitalization, rest and fluids.", "severity": "High", "color": "#F44336"},
    "Malaria": {"emoji": "🦠", "description_en": "Malaria is caused by Plasmodium parasites transmitted by mosquitoes.", "description_hi": "मलेरिया मच्छरों द्वारा फैले परजीवियों से होता है।", "treatment": "Antimalarial drugs, hospitalization in severe cases.", "severity": "High", "color": "#F44336"},
    "Food Poisoning": {"emoji": "🤢", "description_en": "Food poisoning is caused by consuming contaminated food or drink.", "description_hi": "फूड पॉइजनिंग दूषित भोजन से होती है।", "treatment": "Hydration, rest, antibiotics if bacterial cause.", "severity": "Moderate", "color": "#FF9800"},
    "Hypertension": {"emoji": "❤️", "description_en": "High blood pressure that can lead to serious complications.", "description_hi": "उच्च रक्तचाप गंभीर समस्याओं का कारण बन सकता है।", "treatment": "Lifestyle changes, antihypertensive medications.", "severity": "High", "color": "#F44336"},
    "Heart Disease": {"emoji": "💔", "description_en": "Conditions that affect the heart structure and function.", "description_hi": "हृदय की संरचना और कार्य को प्रभावित करने वाली स्थितियां।", "treatment": "Medications, lifestyle changes, surgery in severe cases.", "severity": "Critical", "color": "#B71C1C"},
    "Pneumonia": {"emoji": "🫁", "description_en": "Pneumonia inflames air sacs in one or both lungs.", "description_hi": "निमोनिया फेफड़ों में वायु थैली को सूजन देता है।", "treatment": "Antibiotics, antiviral meds, rest, hospitalization.", "severity": "High", "color": "#F44336"},
    "Bronchitis": {"emoji": "😮", "description_en": "Inflammation of the bronchial tubes that carry air to the lungs.", "description_hi": "ब्रोन्कियल ट्यूब की सूजन।", "treatment": "Rest, fluids, inhalers, antibiotics if bacterial.", "severity": "Moderate", "color": "#FF9800"},
    "Chickenpox": {"emoji": "🔴", "description_en": "Highly contagious viral infection causing an itchy rash.", "description_hi": "चिकनपॉक्स खुजली वाले दाने का कारण बनता है।", "treatment": "Calamine lotion, antihistamines, antiviral drugs.", "severity": "Moderate", "color": "#FF9800"},
    "Strep Throat": {"emoji": "😖", "description_en": "Bacterial infection causing sore throat and fever.", "description_hi": "स्ट्रेप थ्रोट गले में खराश और बुखार का कारण है।", "treatment": "Antibiotics, pain relievers, rest.", "severity": "Moderate", "color": "#FF9800"},
    "Anemia": {"emoji": "🩸", "description_en": "Condition where you lack enough healthy red blood cells.", "description_hi": "एनीमिया में पर्याप्त लाल रक्त कोशिकाएं नहीं होतीं।", "treatment": "Iron supplements, dietary changes.", "severity": "Moderate", "color": "#FF9800"},
    "Hepatitis": {"emoji": "🏥", "description_en": "Inflammation of the liver caused by viral infection.", "description_hi": "हेपेटाइटिस यकृत की सूजन है।", "treatment": "Antiviral medications, rest, avoid alcohol.", "severity": "High", "color": "#F44336"},
    "Gastroenteritis": {"emoji": "🤮", "description_en": "Inflammation of the stomach and intestines.", "description_hi": "गैस्ट्रोएंटेराइटिस पेट और आंतों की सूजन है।", "treatment": "ORS, rest, light diet, probiotics.", "severity": "Mild", "color": "#4CAF50"},
}

@st.cache_resource
def load_model():
    data = pd.read_csv("disease.csv")
    X = data.drop("Disease", axis=1)
    y = data["Disease"]
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, list(X.columns)

model, feature_cols = load_model()

with st.sidebar:
    st.markdown("## 🏥 Disease Predictor")
    st.markdown("---")
    st.markdown("### 📋 About / के बारे में")
    st.markdown("**English:** AI tool that predicts diseases from symptoms using Machine Learning.")
    st.markdown("**हिंदी:** यह AI टूल लक्षणों से बीमारी predict करता है।")
    st.markdown("---")
    st.markdown("### 🎯 SDG-3 Goal")
    st.markdown("Good Health & Well-Being / अच्छा स्वास्थ्य")
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("*Educational only. Always consult a real doctor.*")

st.markdown("""
<div class="main-header">
    <h1>🩺 Disease Prediction System</h1>
    <p>रोग पहचान प्रणाली — Powered by AI & Machine Learning</p>
    <span class="sdg-badge">🌱 SDG-3 · Good Health & Well-Being</span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-box"><h2>16</h2><p>🦠 Diseases</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-box"><h2>15</h2><p>📋 Symptoms</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-box"><h2>AI</h2><p>🤖 Random Forest</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-box"><h2>SDG3</h2><p>🌍 UN Goal</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 📝 Select Symptoms / लक्षण चुनें")
    st.markdown("*Tick all symptoms you have / जो लक्षण हैं वो चुनें:*")
    st.markdown("")

    symptom_labels = {
        "fever":            "🌡️ Fever / बुखार",
        "cough":            "😮 Cough / खांसी",
        "headache":         "🤕 Headache / सिरदर्द",
        "fatigue":          "😴 Fatigue / थकान",
        "body_ache":        "💪 Body Ache / बदन दर्द",
        "nausea":           "🤢 Nausea / मतली",
        "vomiting":         "🤮 Vomiting / उल्टी",
        "diarrhea":         "🚽 Diarrhea / दस्त",
        "rash":             "🔴 Skin Rash / दाने",
        "sore_throat":      "😖 Sore Throat / गले में दर्द",
        "chest_pain":       "💔 Chest Pain / सीने में दर्द",
        "shortness_breath": "😤 Shortness of Breath / सांस में तकलीफ",
        "loss_appetite":    "🍽️ Loss of Appetite / भूख न लगना",
        "sweating":         "💧 Sweating / पसीना",
        "chills":           "🥶 Chills / ठंड लगना"
    }

    user_input = {}
    col_a, col_b = st.columns(2)
    items = list(symptom_labels.items())
    for i, (key, label) in enumerate(items):
        if i % 2 == 0:
            with col_a:
                user_input[key] = 1 if st.checkbox(label, key=key) else 0
        else:
            with col_b:
                user_input[key] = 1 if st.checkbox(label, key=key) else 0

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Disease / बीमारी जांचें")

with right:
    st.markdown("### 📊 Results / परिणाम")

    if predict_btn:
        input_df = pd.DataFrame([user_input])[feature_cols]
        selected_count = sum(user_input.values())

        if selected_count == 0:
            st.warning("⚠️ Please select at least one symptom! / कम से कम एक लक्षण चुनें!")
        else:
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            classes = model.classes_
            confidence = max(proba) * 100
            info = disease_info.get(prediction, {})
            emoji = info.get("emoji", "🏥")
            severity = info.get("severity", "Unknown")
            sev_color = info.get("color", "#e94560")

            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:3rem;">{emoji}</div>
                <div style="color:#a8d8ea;font-size:0.9rem;margin-top:0.5rem;">Predicted Disease / अनुमानित बीमारी</div>
                <div class="result-disease">{prediction}</div>
                <div class="result-confidence">🎯 Confidence: {confidence:.1f}%</div>
                <div style="margin-top:0.8rem;">
                    <span style="background:{sev_color}22;border:1px solid {sev_color};color:{sev_color};padding:0.3rem 1rem;border-radius:30px;font-weight:700;font-size:0.85rem;">
                        ⚡ Severity: {severity}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if info:
                st.markdown(f"""
                <div class="info-card">
                    <h4>ℹ️ About / बीमारी के बारे में</h4>
                    <p style="margin:0;font-size:0.9rem;"><b>EN:</b> {info.get('description_en','')}</p>
                    <p style="margin:0.5rem 0 0 0;font-size:0.9rem;color:#a8d8ea;"><b>हिंदी:</b> {info.get('description_hi','')}</p>
                </div>
                <div class="info-card" style="border-left-color:#FF9800;">
                    <h4 style="color:#FF9800;">💊 Treatment / उपचार</h4>
                    <p style="margin:0;font-size:0.9rem;">{info.get('treatment','Consult a doctor.')}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 📊 Top 5 Predictions")
            top5_idx = np.argsort(proba)[-5:][::-1]
            top5_diseases = [classes[i] for i in top5_idx]
            top5_probs = [proba[i]*100 for i in top5_idx]

            fig = go.Figure(go.Bar(
                x=top5_probs, y=top5_diseases, orientation='h',
                marker=dict(color=top5_probs, colorscale=[[0,'#0f3460'],[0.5,'#e94560'],[1,'#FF6B6B']]),
                text=[f"{p:.1f}%" for p in top5_probs],
                textposition='inside',
                textfont=dict(color='white', size=12)
            ))
            fig.update_layout(
                paper_bgcolor='rgba(15,52,96,0.3)',
                plot_bgcolor='rgba(15,52,96,0.3)',
                font=dict(color='#e0e0e0'),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False),
                margin=dict(l=0,r=10,t=10,b=10),
                height=220
            )
            st.plotly_chart(fig, use_container_width=True)

            selected = [symptom_labels[k] for k, v in user_input.items() if v == 1]
            st.markdown("**✅ Symptoms Selected / चुने गए लक्षण:**")
            chips = " ".join([f'<span class="symptom-chip">{s}</span>' for s in selected])
            st.markdown(chips, unsafe_allow_html=True)

            st.markdown("""
            <div class="warning-card">
                ⚠️ <b>Important:</b> Educational purpose only. Please consult a real doctor.<br>
                <b>महत्वपूर्ण:</b> यह केवल शैक्षिक उद्देश्यों के लिए है। असली डॉक्टर से मिलें।
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#a8d8ea;opacity:0.7;">
            <div style="font-size:5rem;">🩺</div>
            <p style="font-size:1.1rem;margin-top:1rem;">Select symptoms on the left and click Predict</p>
            <p style="font-size:0.95rem;">बाईं तरफ लक्षण चुनें और जांचें पर क्लिक करें</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🗂️ Disease Severity Overview / बीमारियों की गंभीरता")

sev_data = pd.DataFrame([
    {"Disease": d, "Severity": info["severity"], "Emoji": info["emoji"]}
    for d, info in disease_info.items()
])
sev_order = {"Mild": 1, "Moderate": 2, "High": 3, "Critical": 4}
sev_color_map = {"Mild": "#4CAF50", "Moderate": "#FF9800", "High": "#F44336", "Critical": "#B71C1C"}
sev_data["Level"] = sev_data["Severity"].map(sev_order)
sev_data = sev_data.sort_values("Level")

fig2 = px.bar(
    sev_data, x="Disease", y="Level", color="Severity",
    color_discrete_map=sev_color_map,
    text="Emoji",
    labels={"Level": "", "Disease": ""}
)
fig2.update_traces(textfont_size=18, textposition="outside")
fig2.update_layout(
    paper_bgcolor='rgba(15,52,96,0.2)',
    plot_bgcolor='rgba(15,52,96,0.2)',
    font=dict(color='#e0e0e0'),
    xaxis=dict(showgrid=False, tickangle=-30),
    yaxis=dict(showgrid=False, showticklabels=False),
    legend=dict(bgcolor='rgba(0,0,0,0)'),
    margin=dict(t=50, b=50),
    height=320
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
<div style="text-align:center;color:#a8d8ea;font-size:0.85rem;margin-top:1rem;opacity:0.7;">
    Made with ❤️ for SDG-3 · Good Health & Well-Being | AI Disease Prediction System
</div>
""", unsafe_allow_html=True)


