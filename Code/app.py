"""
Campus Placement Prediction & Career Outcome Analytics
Streamlit Dashboard

Run with:
    streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from src.data_generator import generate_placement_data
from src.feature_engineering import add_engineered_features, encode_performance_tier

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Campus Placement Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'placement_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white; padding: 1rem 1.5rem; border-radius: 12px;
        text-align: center; margin-bottom: 1rem;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { margin: 0; opacity: 0.85; font-size: 0.9rem; }
    .placed-badge   { background:#27ae60; color:white; padding:4px 12px;
                      border-radius:20px; font-weight:bold; }
    .notplaced-badge{ background:#e74c3c; color:white; padding:4px 12px;
                      border-radius:20px; font-weight:bold; }
    .section-title  { font-size:1.4rem; font-weight:700; color:#1e3c72;
                      border-bottom:2px solid #2a5298; padding-bottom:6px;
                      margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loaders ──────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df = generate_placement_data(1000)
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    artifacts = {}
    files = {
        'preprocessor'    : 'preprocessor.pkl',
        'best_classifier' : 'best_classifier.pkl',
        'best_regressor'  : 'best_regressor.pkl',
        'salary_scaler'   : 'salary_scaler.pkl',
    }
    for key, fname in files.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
    return artifacts


def models_trained() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, 'best_classifier.pkl'))


# ── Sidebar Navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
    st.title("Campus Placement\nAnalytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔮 Predict Placement", "📊 Analytics Dashboard",
         "📈 Model Performance", "📂 Batch Prediction"],
        index=0
    )
    st.markdown("---")
    st.caption("ML Internship Project")
    st.caption("Built with Streamlit + scikit-learn + XGBoost")


df = load_data()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🎓 Campus Placement Prediction & Career Outcome Analytics")
    st.markdown(
        "A machine learning platform to **predict placement outcomes**, "
        "**estimate expected salary**, and **analyze career trends** across student cohorts."
    )

    if not models_trained():
        st.warning(
            "⚠️ Models not trained yet. Run `python train.py` in the Code folder first, "
            "then refresh this page."
        )

    st.markdown("---")

    # KPI Cards
    total       = len(df)
    placed      = df['placement_status'].sum()
    not_placed  = total - placed
    avg_salary  = df['salary'].mean()
    max_salary  = df['salary'].max()
    place_rate  = placed / total * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h2>{total:,}</h2><p>Total Students</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <h2>{place_rate:.1f}%</h2><p>Placement Rate</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <h2>{avg_salary:.2f} LPA</h2><p>Avg Salary (Placed)</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <h2>{max_salary:.2f} LPA</h2><p>Highest Package</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Dataset Quick View</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    # Placement donut chart
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.pie(
            values=[placed, not_placed],
            names=['Placed', 'Not Placed'],
            hole=0.5,
            color_discrete_sequence=['#27ae60', '#e74c3c'],
            title="Placement Distribution"
        )
        fig.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        salary_data = df[df['salary'].notna()]
        fig2 = px.histogram(
            salary_data, x='salary', nbins=30,
            title="Salary Distribution (Placed Students)",
            color_discrete_sequence=['#2a5298'],
            labels={'salary': 'Salary (LPA)'}
        )
        fig2.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Placement":
    st.title("🔮 Predict Student Placement")

    if not models_trained():
        st.error("Models not found. Run `python train.py` first.")
        st.stop()

    artifacts = load_models()
    preprocessor = artifacts.get('preprocessor')
    clf          = artifacts.get('best_classifier')
    reg          = artifacts.get('best_regressor')

    st.markdown("Fill in the student's profile to get placement prediction and expected salary.")
    st.markdown("---")

    # Mapping from MCA specialization (UI) to model's original MBA values
    MCA_TO_MBA_MAP = {
        "Data Science"        : "Mkt&Fin",
        "Software Engineering": "Mkt&Fin",
        "Information Security": "Mkt&HR",
        "Database Management" : "Mkt&HR",
    }

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics & Skills")
            gender = st.selectbox("Gender", ["Male", "Female", "Transgender"])
            work_experience = st.selectbox("Work Experience", ["No", "Yes"])
            internship_experience = st.selectbox("Internship Experience", ["No", "Yes"])
            communication_skill = st.selectbox("Communication Skill", ["No", "Yes"])
            known_programming_language = st.selectbox(
                "Known Programming Language",
                ["None", "Python", "Java", "C", "C++", "JavaScript", "R", "SQL"]
            )
            mca_specialization = st.selectbox(
                "MCA Specialization",
                ["Data Science", "Software Engineering", "Information Security", "Database Management"]
            )

        with col2:
            st.subheader("Academic — School")
            ssc_percentage  = st.slider("10th % (SSC)", 40.0, 98.0, 70.0, 0.5)
            ssc_board       = st.selectbox("SSC Board", ["Central", "Others"])
            hsc_percentage  = st.slider("12th % (HSC)", 40.0, 98.0, 70.0, 0.5)
            hsc_board       = st.selectbox("HSC Board", ["Central", "Others"])
            hsc_subject     = st.selectbox("HSC Subject", ["Commerce", "Science", "Arts"])

        with col3:
            st.subheader("Academic — College")
            degree_percentage = st.slider("Degree %", 40.0, 98.0, 70.0, 0.5)
            degree_type       = st.selectbox("Degree Type", ["Comm&Mgmt", "Sci&Tech", "Others"])
            etest_percentage  = st.slider("Employability Test %", 30.0, 99.0, 65.0, 0.5)
            aptitude_test_score = st.slider("Aptitude Test Score", 0, 100, 60)
            mca_percentage    = st.slider("MCA %", 40.0, 98.0, 68.0, 0.5)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        # Map MCA specialization selection back to the model's original MBA values
        mba_specialization_mapped = MCA_TO_MBA_MAP.get(mca_specialization, "Mkt&Fin")

        # Model's gender encoder only knows Male/Female; map Transgender → Male as neutral fallback
        gender_mapped = gender if gender in ("Male", "Female") else "Male"

        input_df = pd.DataFrame([{
            'gender'            : gender_mapped,
            'ssc_percentage'    : ssc_percentage,
            'ssc_board'         : ssc_board,
            'hsc_percentage'    : hsc_percentage,
            'hsc_board'         : hsc_board,
            'hsc_subject'       : hsc_subject,
            'degree_percentage' : degree_percentage,
            'degree_type'       : degree_type,
            'work_experience'   : work_experience,
            'etest_percentage'  : etest_percentage,
            'mba_specialization': mba_specialization_mapped,
            'mba_percentage'    : mca_percentage,
        }])

        # Add engineered features (need a dummy row with salary/status for pipeline)
        input_feat = add_engineered_features(input_df)
        input_feat = encode_performance_tier(input_feat)

        # Preprocess
        X_input = preprocessor.transform(input_feat)

        # Predict placement
        placement_prob = clf.predict_proba(X_input)[0][1]
        placement_pred = int(placement_prob >= 0.5)

        st.markdown("---")
        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            label = "PLACED" if placement_pred == 1 else "NOT PLACED"
            badge_cls = "placed-badge" if placement_pred == 1 else "notplaced-badge"
            st.markdown(f"**Placement Outcome:** <span class='{badge_cls}'>{label}</span>",
                        unsafe_allow_html=True)

        with res_col2:
            st.metric("Placement Probability", f"{placement_prob:.1%}")

        with res_col3:
            if placement_pred == 1 and reg is not None:
                sal_input = preprocessor.transform(input_feat)
                salary_pred = reg.predict(sal_input)[0]
                st.metric("Predicted Salary", f"₹ {salary_pred:.2f} LPA")
            else:
                st.metric("Predicted Salary", "N/A (Not Placed)")

        # Gauge chart for probability
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=placement_prob * 100,
            title={'text': "Placement Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#27ae60" if placement_prob >= 0.5 else "#e74c3c"},
                'steps': [
                    {'range': [0, 40],  'color': '#fde8e8'},
                    {'range': [40, 60], 'color': '#fef9e7'},
                    {'range': [60, 100],'color': '#e8f8f0'},
                ],
                'threshold': {'line': {'color': 'black', 'width': 3}, 'value': 50}
            }
        ))
        fig.update_layout(height=300, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # Profile strengths/weaknesses
        st.markdown('<div class="section-title">Profile Analysis</div>', unsafe_allow_html=True)
        academic_avg = np.mean([ssc_percentage, hsc_percentage, degree_percentage, mca_percentage])
        recommendations = []
        if degree_percentage < 60:
            recommendations.append("📚 Degree percentage is below 60% — focus on improving academic performance.")
        if etest_percentage < 60:
            recommendations.append("💡 Employability test score is low — practice aptitude and technical tests.")
        if aptitude_test_score < 50:
            recommendations.append("🧠 Aptitude test score is below 50 — practice quantitative and logical reasoning.")
        if work_experience == 'No' and internship_experience == 'No':
            recommendations.append("💼 No work/internship experience — consider internships or live projects.")
        elif work_experience == 'No' and internship_experience == 'Yes':
            recommendations.append("💼 Internship experience noted! Pursue full-time roles to further strengthen your profile.")
        if communication_skill == 'No':
            recommendations.append("🗣️ Communication skills need improvement — join clubs or practice presentations.")
        if known_programming_language == 'None':
            recommendations.append("💻 Learn a programming language (Python/Java) to boost your employability.")
        if mca_percentage < 65:
            recommendations.append("📖 MCA percentage is below 65% — aim for consistent academic performance.")
        if not recommendations:
            recommendations.append("✅ Strong profile! Keep maintaining your academic and professional growth.")
        for rec in recommendations:
            st.info(rec)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.title("📊 Career Outcome Analytics Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["By Category", "Academic Analysis", "Salary Insights", "Correlation"]
    )

    # ── Tab 1: Category Analysis ──────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Placement Rate by Category</div>',
                    unsafe_allow_html=True)

        cat_options = ['gender', 'hsc_subject', 'degree_type',
                       'mba_specialization', 'work_experience', 'ssc_board']
        selected_cat = st.selectbox("Select Category", cat_options)

        grp = df.groupby(selected_cat)['placement_status'].agg(['sum', 'count']).reset_index()
        grp.columns = [selected_cat, 'placed', 'total']
        grp['placement_rate'] = (grp['placed'] / grp['total'] * 100).round(1)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(
                grp, x=selected_cat, y='placement_rate',
                title=f"Placement Rate by {selected_cat.replace('_', ' ').title()}",
                color='placement_rate', color_continuous_scale='Blues',
                text='placement_rate', labels={'placement_rate': 'Placement Rate (%)'}
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = px.bar(
                grp, x=selected_cat, y=['placed', 'total'],
                title=f"Placed vs Total by {selected_cat.replace('_', ' ').title()}",
                barmode='group',
                color_discrete_sequence=['#27ae60', '#95a5a6'],
                labels={'value': 'Count', 'variable': 'Status'}
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Academic Analysis ───────────────────────────────────
    with tab2:
        st.markdown('<div class="section-title">Academic Performance vs Placement</div>',
                    unsafe_allow_html=True)

        academic_cols = ['ssc_percentage', 'hsc_percentage',
                         'degree_percentage', 'etest_percentage', 'mba_percentage']
        selected_acad = st.selectbox("Select Academic Metric", academic_cols)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.box(
                df, x='placement_status', y=selected_acad,
                color='placement_status',
                color_discrete_map={0: '#e74c3c', 1: '#27ae60'},
                labels={'placement_status': 'Placement (0=No, 1=Yes)',
                        selected_acad: selected_acad.replace('_', ' ').title()},
                title=f"{selected_acad.replace('_', ' ').title()} by Placement Status"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = px.histogram(
                df, x=selected_acad, color='placement_status',
                barmode='overlay', opacity=0.7,
                color_discrete_map={0: '#e74c3c', 1: '#27ae60'},
                title=f"Distribution of {selected_acad.replace('_', ' ').title()}",
                labels={'placement_status': 'Placed'}
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Salary Insights ─────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">Salary Insights (Placed Students)</div>',
                    unsafe_allow_html=True)
        placed_df = df[df['placement_status'] == 1].copy()

        col_a, col_b = st.columns(2)
        with col_a:
            grp_sal = placed_df.groupby('mba_specialization')['salary'].mean().reset_index()
            fig = px.bar(
                grp_sal, x='mba_specialization', y='salary',
                title="Avg Salary by MBA Specialization",
                color='mba_specialization',
                color_discrete_sequence=['#2a5298', '#e74c3c'],
                labels={'salary': 'Avg Salary (LPA)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            grp_sal2 = placed_df.groupby('gender')['salary'].mean().reset_index()
            fig2 = px.bar(
                grp_sal2, x='gender', y='salary',
                title="Avg Salary by Gender",
                color='gender',
                color_discrete_sequence=['#3498db', '#e91e8c'],
                labels={'salary': 'Avg Salary (LPA)'}
            )
            st.plotly_chart(fig2, use_container_width=True)

        grp_exp = placed_df.groupby('work_experience')['salary'].mean().reset_index()
        col_c, col_d = st.columns(2)
        with col_c:
            fig3 = px.bar(
                grp_exp, x='work_experience', y='salary',
                title="Avg Salary: Work Experience Effect",
                color='work_experience',
                color_discrete_sequence=['#27ae60', '#e67e22'],
                labels={'salary': 'Avg Salary (LPA)'}
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            fig4 = px.scatter(
                placed_df, x='mba_percentage', y='salary',
                color='mba_specialization',
                title="MBA % vs Salary",
                labels={'mba_percentage': 'MBA %', 'salary': 'Salary (LPA)'}
            )
            # Manual trendline (no statsmodels needed)
            z = np.polyfit(placed_df['mba_percentage'], placed_df['salary'], 1)
            x_line = np.linspace(placed_df['mba_percentage'].min(), placed_df['mba_percentage'].max(), 100)
            fig4.add_scatter(x=x_line, y=np.polyval(z, x_line),
                             mode='lines', name='Trend', line=dict(color='black', dash='dash'))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 4: Correlation ────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-title">Feature Correlation Heatmap</div>',
                    unsafe_allow_html=True)
        num_df = df[['ssc_percentage', 'hsc_percentage', 'degree_percentage',
                     'etest_percentage', 'mba_percentage', 'placement_status', 'salary']].copy()
        corr = num_df.corr()

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, square=True, linewidths=0.5)
        ax.set_title('Correlation Matrix', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Comparison")

    if not models_trained():
        st.error("Models not trained. Run `python train.py` first.")
        st.stop()

    clf_csv = os.path.join(MODEL_DIR, 'classifier_results.csv')
    reg_csv = os.path.join(MODEL_DIR, 'regressor_results.csv')

    if os.path.exists(clf_csv):
        st.markdown('<div class="section-title">Classification Results</div>',
                    unsafe_allow_html=True)
        clf_df = pd.read_csv(clf_csv, index_col='Model')
        st.dataframe(clf_df.style.highlight_max(color='lightgreen', axis=0), use_container_width=True)

        fig = px.bar(
            clf_df.reset_index(), x='Model', y=['Accuracy', 'F1 Score', 'ROC-AUC'],
            barmode='group',
            title="Classification Metrics — All Models",
            color_discrete_sequence=['#2a5298', '#27ae60', '#e67e22']
        )
        st.plotly_chart(fig, use_container_width=True)

    if os.path.exists(reg_csv):
        st.markdown('<div class="section-title">Regression Results (Salary)</div>',
                    unsafe_allow_html=True)
        reg_df = pd.read_csv(reg_csv, index_col='Model')
        st.dataframe(reg_df.style.highlight_max(subset=['R²'], color='lightgreen'), use_container_width=True)

        fig2 = px.bar(
            reg_df.reset_index(), x='Model', y='R²',
            title="R² Score — Salary Regressors",
            color='R²', color_continuous_scale='Blues',
            text='R²'
        )
        fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

    # Show saved plots
    plots_dir = os.path.join(BASE_DIR, 'plots')
    if os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        if plot_files:
            st.markdown('<div class="section-title">Saved Plots</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, fname in enumerate(sorted(plot_files)):
                with cols[i % 2]:
                    st.image(os.path.join(plots_dir, fname), caption=fname.replace('_', ' ').replace('.png', ''))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Prediction")
    st.markdown("Upload a CSV with student data to get placement predictions for all students.")

    if not models_trained():
        st.error("Models not trained. Run `python train.py` first.")
        st.stop()

    artifacts    = load_models()
    preprocessor = artifacts.get('preprocessor')
    clf          = artifacts.get('best_classifier')
    reg          = artifacts.get('best_regressor')

    # Sample template download
    sample_cols = [
        'gender', 'ssc_percentage', 'ssc_board', 'hsc_percentage', 'hsc_board',
        'hsc_subject', 'degree_percentage', 'degree_type', 'work_experience',
        'etest_percentage', 'mba_specialization', 'mba_percentage'
    ]
    sample = pd.DataFrame([{
        'gender': 'Male', 'ssc_percentage': 72.0, 'ssc_board': 'Central',
        'hsc_percentage': 68.0, 'hsc_board': 'Others', 'hsc_subject': 'Commerce',
        'degree_percentage': 65.0, 'degree_type': 'Comm&Mgmt', 'work_experience': 'No',
        'etest_percentage': 70.0, 'mba_specialization': 'Mkt&Fin', 'mba_percentage': 67.0
    }])
    st.download_button(
        "⬇️ Download Sample CSV Template",
        data=sample.to_csv(index=False),
        file_name='student_template.csv',
        mime='text/csv'
    )

    uploaded = st.file_uploader("Upload Student CSV", type=['csv'])
    if uploaded:
        upload_df = pd.read_csv(uploaded)
        st.markdown(f"**Uploaded:** {len(upload_df)} students")
        st.dataframe(upload_df.head(), use_container_width=True)

        with st.spinner("Running predictions..."):
            feat_df = add_engineered_features(upload_df)
            feat_df = encode_performance_tier(feat_df)
            X_batch = preprocessor.transform(feat_df)

            probs    = clf.predict_proba(X_batch)[:, 1]
            preds    = (probs >= 0.5).astype(int)
            salaries = np.where(preds == 1, reg.predict(X_batch), np.nan)

            result_df = upload_df.copy()
            result_df['placement_probability'] = np.round(probs, 4)
            result_df['placement_prediction']  = preds
            result_df['placement_label']       = np.where(preds == 1, 'Placed', 'Not Placed')
            result_df['predicted_salary_lpa']  = np.round(salaries, 2)

        st.success(f"✅ Done! Predicted {preds.sum()} placements out of {len(preds)} students.")
        st.dataframe(result_df, use_container_width=True)

        st.download_button(
            "⬇️ Download Predictions CSV",
            data=result_df.to_csv(index=False),
            file_name='placement_predictions.csv',
            mime='text/csv'
        )
