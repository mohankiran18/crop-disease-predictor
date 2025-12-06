import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import datetime
from firebase_admin import firestore
import google.generativeai as genai

# Import our modular logic
import utils

# ============ PAGE CONFIG ============
st.set_page_config(page_title="Crop disease Predictor", page_icon="🌐", layout="wide")

# Apply CSS from utils
st.markdown(utils.get_custom_css(), unsafe_allow_html=True)

# Initialize DB
db = utils.get_db()

# ============ SIDEBAR ============
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10543/10543161.png", width=80)
    st.markdown(
        "<h3 style='text-align: center; color: #064E3B;'>Crop Disease Predictor</h3>",
        unsafe_allow_html=True
    )
    app_mode = option_menu(
        menu_title=None,
        options=["Dashboard", "Diagnosis", "History", "Analytics", "AI Expert"],
        icons=["grid-fill", "camera-fill", "clock-fill", "bar-chart-fill", "chat-dots-fill"],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "transparent"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#f0f2f6",
            },
            "nav-link-selected": {
                "background-color": "#ECFDF5",
                "color": "#059669",
                "font-weight": "600",
            },
        },
    )
    st.markdown("---")
    if utils.check_backend_status():
        st.success("Backend Online 🟢")
    else:
        st.error("Backend Offline 🔴")
    
    # ============ CHATBOT (Sidebar Location) ============
    # Placed here so the empty space is hidden in the menu, 
    # not on the main dashboard.
    st.markdown("### 💬 AI Support")
    utils.embed_chatbot()

# ============ PAGES ============

# ---------- DASHBOARD ----------
if app_mode == "Dashboard":
    today = datetime.datetime.now().strftime("%A, %d %B %Y")
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-text">
                <h1>Hello, Farmer 👋</h1>
                <p>Monitor your crops, detect diseases, and get expert advice.</p>
            </div>
            <div class="hero-date">📅 {today}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="stCard"><h3>🧠 Dual-Core AI</h3>'
            '<p>Combines your <b>Custom CNN Model</b> for accurate disease '
            'identification with <b>Artificial intelligence</b> for expert treatment advice.</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="stCard"><h3>🌍 Regional Support</h3>'
            '<p>Get advice in <b>English, Hindi, Telugu, and Tamil</b>. '
            'We speak your language to help you farm better.</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="stCard"><h3>📄 Instant PDF</h3>'
            '<p>Download <b>professional reports</b> for your records instantly. '
            'Useful for insurance and farm auditing.</p></div>',
            unsafe_allow_html=True,
        )

# ---------- DIAGNOSIS ----------
elif app_mode == "Diagnosis":
    st.markdown('<div class="header-pill">🔬 Start Diagnosis</div>', unsafe_allow_html=True)
    r1, r2 = st.columns([1, 1.5])

    with r1:
        st.markdown(
            '<div class="header-pill" style="font-size:1rem; padding:8px 15px;">📤 Upload Image</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Select Leaf Photo", type=["jpg", "png"], label_visibility="collapsed"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="header-pill" style="font-size:1rem; padding:8px 15px;">🗣️ Select Language</div>',
            unsafe_allow_html=True,
        )
        lang = st.selectbox(
            "Select Language",
            ["English", "Hindi", "Telugu", "Tamil"],
            label_visibility="collapsed",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("🔍 Analyze Crop")

    with r2:
        st.markdown(
            '<div style="background-color: rgba(255, 255, 255, 0.95); padding: 10px; '
            'border-radius: 10px 10px 0 0; border-bottom: 1px solid #eee;">'
            '<h4 style="margin:0; color: #064E3B;">🖼️ Image Preview</h4></div>',
            unsafe_allow_html=True,
        )
        if uploaded:
            st.image(uploaded, use_container_width=True)
        else:
            st.markdown(
                '<div style="background-color: rgba(255, 255, 255, 0.9); height: 300px; '
                'border-radius: 0 0 10px 10px; display: flex; align-items: center; '
                'justify-content: center; color: #6B7280; border: 2px dashed #E5E7EB;">'
                '<div style="text-align:center;"><span style="font-size: 3rem;">📸</span>'
                '<p>No image uploaded</p></div></div>',
                unsafe_allow_html=True,
            )

    if analyze and uploaded:
        with st.spinner("🤖 Analyzing leaf patterns..."):
            disease, confidence = utils.predict_via_api(uploaded)

            if disease == "Connection Error":
                st.error("🔴 Connection Error. Is the Backend running?")
            elif "Error" in str(disease):
                st.error(f"🔴 Prediction Failed: {disease}")
            else:
                details = utils.get_disease_details_gemini(disease, lang)
                sections = utils.parse_sections(details)

                if db:
                    db.collection("diagnoses").add(
                        {
                            "disease": disease,
                            "confidence": confidence,
                            "timestamp": datetime.datetime.utcnow(),
                            "details": details,
                        }
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="background-color: #ECFDF5; padding: 20px; border-radius: 12px; 
                         border: 1px solid #059669; text-align: center; margin-bottom: 20px;">
                        <h2 style="color: #064E3B; margin:0;">✅ Detected: {disease}</h2>
                        <p style="color: #047857; margin:5px 0 0 0; font-weight:600;">
                            Confidence: {confidence*100:.1f}%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                t1, t2, t3, t4 = st.tabs(
                    ["📌 Causes", "🦠 Symptoms", "🛡 Precautions", "💊 Treatments"]
                )

                def card(text):
                    return f'<div class="stCard">{text}</div>'

                with t1:
                    st.markdown(card(sections["Causes"]), unsafe_allow_html=True)
                with t2:
                    st.markdown(card(sections["Symptoms"]), unsafe_allow_html=True)
                with t3:
                    st.markdown(card(sections["Precautions"]), unsafe_allow_html=True)
                with t4:
                    st.markdown(card(sections["Treatments"]), unsafe_allow_html=True)

                clean_filename = f"{disease.replace(' ', '_')}_Report.pdf"
                pdf = utils.create_pdf_report(disease, details)
                st.download_button(
                    "📥 Download PDF Report",
                    pdf,
                    file_name=clean_filename,
                    mime="application/pdf",
                )

# ---------- HISTORY ----------
elif app_mode == "History":
    st.markdown('<div class="header-pill">📜 History Log</div>', unsafe_allow_html=True)
    if db:
        try:
            docs = (
                db.collection("diagnoses")
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .stream()
            )
            data = []
            for d in docs:
                dd = d.to_dict()
                ts = dd.get("timestamp")
                date_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "N/A"
                data.append(
                    {
                        "ID": d.id,
                        "Select": False,
                        "Date": date_str,
                        "Disease": dd.get("disease", "Unknown"),
                        "Confidence": f"{dd.get('confidence', 0)*100:.1f}%",
                    }
                )

            if data:
                edited_df = st.data_editor(
                    pd.DataFrame(data),
                    column_config={
                        "ID": None,
                        "Select": st.column_config.CheckboxColumn(
                            "Select", width="small"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                c1, _, c3 = st.columns([1, 2, 1])
                with c1:
                    if st.button(
                        "🗑️ Delete Selected",
                        type="primary",
                        use_container_width=True,
                    ):
                        to_delete = edited_df[edited_df["Select"] == True]
                        if not to_delete.empty:
                            for _, row in to_delete.iterrows():
                                db.collection("diagnoses").document(row["ID"]).delete()
                            st.toast(f"✅ Deleted {len(to_delete)} records!")
                            st.rerun()
                        else:
                            st.toast("⚠️ Select rows first!", icon="⚠️")
                with c3:
                    with st.expander("🔥 Clear History"):
                        if st.button(
                            "Confirm Delete All",
                            type="secondary",
                            use_container_width=True,
                        ):
                            for drow in data:
                                db.collection("diagnoses").document(
                                    drow["ID"]
                                ).delete()
                            st.toast("🔥 History Cleared!", icon="✅")
                            st.rerun()
            else:
                st.info("No records found.")
        except Exception:
            st.error("Error fetching history.")
    else:
        st.warning("Database not connected")

# ---------- ANALYTICS ----------
elif app_mode == "Analytics":
    st.markdown('<div class="header-pill">📊 Farm Analytics</div>', unsafe_allow_html=True)
    if db:
        docs = list(db.collection("diagnoses").stream())
        total_scans = len(docs)
        if total_scans > 0:
            counts = {}
            for d in docs:
                n = d.to_dict().get("disease", "Unknown")
                counts[n] = counts.get(n, 0) + 1
            df = pd.DataFrame(
                {"Disease": list(counts.keys()), "Count": list(counts.values())}
            )
            most_common = max(counts, key=counts.get)

            m1, m2, m3 = st.columns(3)

            def metric_card(title, value, icon):
                st.markdown(
                    f"""
                    <div style="background-color:rgba(255,255,255,0.95);padding:20px;
                         border-radius:12px;border:1px solid #E5E7EB;text-align:center;">
                        <div style="font-size:2rem;">{icon}</div>
                        <div style="color:#6B7280;font-weight:600;">{title}</div>
                        <div style="color:#064E3B;font-size:1.8rem;font-weight:700;">
                            {value}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m1:
                metric_card("Total Scans", total_scans, "📷")
            with m2:
                metric_card("Most Detected", most_common, "⚠️")
            with m3:
                metric_card("Unique Diseases", len(counts), "🦠")

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    '<div style="background-color:rgba(255,255,255,0.95);padding:20px;'
                    'border-radius:12px;border:1px solid #E5E7EB;height:100%;">'
                    '<h4 style="color:#064E3B;text-align:center;">Disease Distribution</h4>',
                    unsafe_allow_html=True,
                )
                fig = px.pie(
                    df,
                    values="Count",
                    names="Disease",
                    hole=0.6,
                    color_discrete_sequence=[
                        "#059669",
                        "#10B981",
                        "#34D399",
                        "#6EE7B7",
                    ],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=0, b=0, l=0, r=0),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(
                    '<div style="background-color:rgba(255,255,255,0.95);padding:20px;'
                    'border-radius:12px;border:1px solid #E5E7EB;height:100%;">'
                    '<h4 style="color:#064E3B;text-align:center;">Frequency Analysis</h4>',
                    unsafe_allow_html=True,
                )
                fig2 = px.bar(df, x="Disease", y="Count")
                fig2.update_traces(marker_color="#059669")
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=0, b=0, l=0, r=0),
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No data yet.")
    else:
        st.warning("Database not connected")

# ---------- AI EXPERT ----------
elif app_mode == "AI Expert":
    st.markdown('<div class="header-pill">🤖 Agri-Consultant</div>', unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown(
            '<div style="background-color:rgba(255,255,255,0.85);padding:20px;'
            'border-radius:12px;margin-bottom:20px;text-align:center;">'
            "<h3 style='color:#064E3B;margin:0;'>👋 How can I help?</h3>"
            "<p style='color:#4B5563;'>Select a topic below.</p></div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🌽 Pest Control", use_container_width=True):
                st.session_state.messages.append(
                    {"role": "user", "content": "How to control corn pests?"}
                )
                st.rerun()
        with c2:
            if st.button("🌧️ Weather Impact", use_container_width=True):
                st.session_state.messages.append(
                    {"role": "user", "content": "Effect of rain on wheat?"}
                )
                st.rerun()
        with c3:
            if st.button("🧪 Soil Health", use_container_width=True):
                st.session_state.messages.append(
                    {"role": "user", "content": "Fertilizer for acidic soil?"}
                )
                st.rerun()

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    # Updated Prompt with Language Rules
                    system_rules = (
                        "You are an expert Agricultural Consultant. "
                        "Your goal is to help farmers and users with accurate farming advice.\n\n"
                        "CRITICAL INSTRUCTIONS FOR LANGUAGE ADAPTATION:\n"
                        "1. DETECT the language of the user's question.\n"
                        "2. ADAPT your response based on the following rules:\n"
                        "   - IF English: Reply in clear, professional English.\n"
                        "   - IF Hindi or Hinglish: Reply in friendly Hinglish. Start with 'Ram Ram, Kisan bhai'.\n"
                        "   - IF Telugu: Reply in Telugu script. Start with 'Namaskaram Raitu bidda'.\n"
                        "   - IF Tamil: Reply in Tamil script. Start with 'Vanakkam Vivasaayi Thozhagale'.\n"
                        "3. ALWAYS match the user's language and script."
                    )

                    # Combine the rules with the user's question
                    prompt = f"{system_rules}\n\nQuestion: {st.session_state.messages[-1]['content']}"
                    
                    resp = model.generate_content(prompt)
                    answer = resp.text
                except Exception as e:
                    answer = f"Sorry, I could not answer due to an error: {e}"
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

    if p := st.chat_input("Ask about farming..."):
        st.session_state.messages.append({"role": "user", "content": p})
        st.rerun()