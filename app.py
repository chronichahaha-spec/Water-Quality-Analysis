import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

#Page Setting
st.set_page_config(
    page_title="Water Quality XAI Explanation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1a237e;
        font-size: 2.8rem;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        border-left: 6px solid #0d47a1;
    }
    .role-tab {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    .section-header {
        color: #0d47a1;
        border-left: 5px solid #2196f3;
        padding-left: 15px;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        font-size: 1.8rem;
    }
    .feature-card {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .info-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        margin: 20px 0;
        border-radius: 0 5px 5px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 15px;
        margin: 20px 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

#Main Title
st.markdown('<div class="main-title">📊 Water Quality XAI Explanation Dashboard</div>', unsafe_allow_html=True)

#Stakeholder Roles
st.markdown("### Please select you role:")
tab1, tab2, tab3 = st.tabs(["📊 **Environment Regulator**", "🏭 **Water Supplyment Company**", "👨‍👩‍👧‍👦 **Resident**"])

#Sidebar
with st.sidebar:
    st.markdown("### XAI Explanation System Information")
    st.markdown("""
    **Model**: Random Forest  
    **XAI Method**: SHAP DiCE 
    **Data**: Water Potability Dataset from Kaggle
    """)
    
    st.markdown("---")
    st.markdown("### Feature Envolved In Model Prediction")
    st.markdown("""
    - **PH
    - **Hardness
    - **Solids
    - **Chloramines
    - **Sulfate
    - **Conductivity
    - **Organic_carbon
    - **Trihalomethanes
    - **Turbidity
    """)

# ==================== Load Data/Train Model ====================
@st.cache_resource
def load_data_and_train():
    
    #Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    #Step1: Load Data
    status_text.text("Step1/4: Load Dataset...")
    progress_bar.progress(25)
    
    try:
        df = pd.read_csv("data/water_potability.csv")
    except FileNotFoundError:
        st.error("Fail to search data file: data/water_potability.csv")
        st.stop()
    
    #Step2: Preprocess Data
    status_text.text("Step2/4: Preprocess Data...")
    progress_bar.progress(50)
    
    #Fill Nan Value
    for col in ['ph', 'Sulfate', 'Trihalomethanes']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    
    feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    X = df[feature_names].copy()
    y = df['Potability'].copy()
    
    # Step3: Train Model
    status_text.text("Step3/4: Train Model...")
    progress_bar.progress(75)
    
    #Train/Test Set Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    #Oversample
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    #Train Random Forest Data
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    #Model Performance
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    #Step4: Compute SHAP Value
    status_text.text("Step4/4: Compute SHAP Value...")
    progress_bar.progress(95)
    
    explainer = shap.TreeExplainer(rf_model)
    
    #SHAP Value
    shap_values = explainer.shap_values(X_test)
    
    #SHAP std
    shap_std = {}
    if shap_values is not None and len(shap_values) > 1:
        shap_class1 = shap_values[:,:,1]  #Class 1: Potable
        for i, feature in enumerate(feature_names):
            shap_std[feature] = np.std(shap_class1[:, i])
    
    status_text.text("Done Model Training")
    progress_bar.progress(100)
    
    #Remove Progress Bar
    progress_bar.empty()
    status_text.empty()
    
    return {
        'model': rf_model,
        'explainer': explainer,
        'shap_values': shap_values,
        'X_test': X_test,
        'X_train': X_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_names': feature_names,
        'shap_std': shap_std,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }

#Load Data/Train Model
with st.spinner('Initialize Model...'):
    data_dict = load_data_and_train()

rf_model = data_dict['model']
explainer = data_dict['explainer']
shap_values = data_dict['shap_values']
X_test = data_dict['X_test']
y_test = data_dict['y_test']
feature_names = data_dict['feature_names']
shap_std = data_dict['shap_std']
metrics = data_dict['metrics']

# ==================== Environment Regulator ====================
with tab1:
    st.markdown('<div class="section-header">📊 Environment Regulator - XAI Dashboard</div>', unsafe_allow_html=True)
    
    #Show RF Model Performance
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("F1 Score", f"{metrics['f1']:.3f}")
    
    st.markdown("---")
    
    #P1 SHAP Summary Plot
    st.markdown('<div class="section-header">1. Global Feature Importance Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">This Plot Shows Global Feature Importance Rank (Ascending Order)</div>', unsafe_allow_html=True)
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    shap.summary_plot(
        shap_values[:,:,1],  #Class 1: Water Potable
        X_test,
        feature_names=feature_names,
        show=False,
        plot_type="dot"
    )
    plt.title("SHAP Feature Importance Summary Plot", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1)
    
    #SHAP STD Plot
    st.markdown('<div class="section-header">SHAP Standard Deviation Plot</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box">Higher STD Value implies Higher Feature Importance to Potability Prediction</div>', unsafe_allow_html=True)
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    features = list(shap_std.keys())
    std_values = list(shap_std.values())
    
    #Sort STD Values (ascending sequence)
    sorted_indices = np.argsort(std_values)
    features_sorted = [features[i] for i in sorted_indices]
    values_sorted = [std_values[i] for i in sorted_indices]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features_sorted)))
    bars = ax2.barh(features_sorted, values_sorted, color=colors)
    
    #Insert Label to Bar Chart
    for bar, value in zip(bars, values_sorted):
        width = bar.get_width()
        ax2.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left', va='center')
    
    ax2.set_xlabel('SHAP Standard Deviation', fontsize=12)
    ax2.set_title('Feature Importance Rank to Potability(Class 1)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig2)
    
    #SHAP Dependency Plot
    st.markdown('<div class="section-header">3. Relationship between Feature Value and Prediction (Feature Dependency)</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Click to view impact(dependency) of each features to prediction</div>', unsafe_allow_html=True)
    
    #Expand All/Collapse All Button
    col_expand1, col_expand2 = st.columns(2)
    with col_expand1:
        if st.button("🔼 Expand All", use_container_width=True):
            st.session_state.expand_all = False
    with col_expand2:
        if st.button("🔽 Collapse All", use_container_width=True):
            st.session_state.expand_all = True
    
    #Initialize Session State
    if 'expand_all' not in st.session_state:
        st.session_state.expand_all = False
    
    # 9个特征的依赖图
    feature_descriptions = {
        'ph': '酸碱度是水质的基本指标，影响水处理效果和管道腐蚀。',
        'Hardness': '硬度主要反映钙镁离子含量，过高会导致结垢问题。',
        'Solids': '总溶解固体反映水中矿物质含量，影响口感和健康。',
        'Chloramines': '消毒副产物，浓度需严格控制以保障安全。',
        'Sulfate': '硫酸盐过高可能引起肠胃不适，需监测控制。',
        'Conductivity': '电导率反映水中离子总量，间接指示污染程度。',
        'Organic_carbon': '有机碳是微生物的营养源，过高可能滋生细菌。',
        'Trihalomethanes': '重要消毒副产物，有潜在致癌风险。',
        'Turbidity': '浊度反映水中悬浮物含量，影响消毒效果。'
    }
    
    #Create Folding Subfield for Each Features
    for i, feature in enumerate(feature_names):
        with st.expander(f"**{feature}** - {feature_descriptions.get(feature, '')}", 
                        expanded=st.session_state.expand_all):
            
            st.markdown(f'<div class="feature-card">', unsafe_allow_html=True)
            
            #Split into left/right column
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                #SHAP Dependency Plot
                fig_dep, ax_dep = plt.subplots(figsize=(8, 4))
                shap.dependence_plot(
                    feature,
                    shap_values[:,:,1],
                    X_test,
                    feature_names=feature_names,
                    ax=ax_dep,
                    show=False
                )
                ax_dep.set_title(f'{feature} SHAP Dependency Plot', fontsize=12, fontweight='bold')
                ax_dep.set_xlabel(feature, fontsize=10)
                ax_dep.set_ylabel('SHAP Value (Potability/Class 1 Prediction)', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_dep)
            
            with col_right:
                # 显示特征统计信息
                st.markdown("##### 特征统计")
                st.write(f"**均值**: {X_test[feature].mean():.2f}")
                st.write(f"**标准差**: {X_test[feature].std():.2f}")
                st.write(f"**范围**: [{X_test[feature].min():.2f}, {X_test[feature].max():.2f}]")
                st.write(f"**SHAP波动**: {shap_std.get(feature, 0):.4f}")
                
                # 监管建议
                st.markdown("##### 监管建议")
                if feature in ['Trihalomethanes', 'Chloramines']:
                    st.info("需严格监控，定期检测")
                elif feature in ['ph', 'Turbidity']:
                    st.warning("需保持稳定，避免波动")
                else:
                    st.success("常规监测指标")
            
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== Water Supplyment Company ====================
with tab2:
    st.markdown('<div class="section-header">🏭 Water Supplyment Company View</div>', unsafe_allow_html=True)
    
    #Split into left/right column
    col_input2, col_viz2 = st.columns([1, 2])
    
    with col_input2:
        st.markdown("### 📝 Please Input Your Water Sample Data")
        #st.markdown('<div class="info-box">System will analyze the input data and give advise</div>', unsafe_allow_html=True)
        
        if 'company_inputs' not in st.session_state:
            st.session_state.company_inputs = {
                'ph': 7.0,
                'Hardness': 150.0,
                'Solids': 20000.0,
                'Chloramines': 4.0,
                'Sulfate': 250.0,
                'Conductivity': 400.0,
                'Organic_carbon': 10.0,
                'Trihalomethanes': 50.0,
                'Turbidity': 3.0
            }
        
        #User Input Module
        with st.form("company_water_quality_form"):
            
            ph_value2 = st.slider("**ph**", 0.0, 14.0, st.session_state.company_inputs['ph'], 0.1,
                                key='ph2')
            hardness_value2 = st.slider("**Hardness (mg/L)**", 47.0, 323.0, st.session_state.company_inputs['Hardness'], 1.0,
                                        key='hardness2')
            solids_value2 = st.slider("**Solids (mg/L)**", 320.0, 61227.0, st.session_state.company_inputs['Solids'], 100.0,
                                        key='solids2')
            chloramines_value2 = st.slider("**Chloramines (mg/L)**", 0.35, 13.0, st.session_state.company_inputs['Chloramines'], 0.1,
                                          key='chloramines2')
            sulfate_value2 = st.slider("**Sulfate (mg/L)**", 129.0, 481.0, st.session_state.company_inputs['Sulfate'], 1.0,
                                     key='sulfate2')
            conductivity_value2 = st.slider("**Conductivity (μS/cm)**", 181.0, 753.0, st.session_state.company_inputs['Conductivity'], 1.0,
                                          key='conductivity2')
            organic_carbon_value2 = st.slider("**Organic_carbon (mg/L)**", 2.2, 28.0, st.session_state.company_inputs['Organic_carbon'], 0.1,
                                             key='organic_carbon2')
            trihalomethanes_value2 = st.slider("**Trihalomethanes (μg/L)**", 0.7, 124.0, st.session_state.company_inputs['Trihalomethanes'], 0.1,
                                              key='trihalomethanes2')
            turbidity_value2 = st.slider("**Turbidity (NTU)**", 1.45, 6.74, st.session_state.company_inputs['Turbidity'], 0.1,
                                        key='turbidity2')
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                #Submission Button
                submitted2 = st.form_submit_button("🔍 Analyze Water Potablity", type="primary", use_container_width=True)
            with col_btn2:
                #DiCE Advice
                get_dice = st.form_submit_button("🔄 Access DiCE Advise", type="secondary", use_container_width=True)
    
    with col_viz2:
        st.markdown("### 📊 Prediction Result Analysis/Optimization Advise")
        
        #Check Input Submission
        if submitted2 or get_dice:
            
            st.session_state.company_inputs = {
                'ph': ph_value2,
                'Hardness': hardness_value2,
                'Solids': solids_value2,
                'Chloramines': chloramines_value2,
                'Sulfate': sulfate_value2,
                'Conductivity': conductivity_value2,
                'Organic_carbon': organic_carbon_value2,
                'Trihalomethanes': trihalomethanes_value2,
                'Turbidity': turbidity_value2
            }
            
            #Convert Input to Data Frame
            company_input = pd.DataFrame([st.session_state.company_inputs])
            
            with st.spinner("Analyzing Water Potability"):
                #Prediction
                proba2 = rf_model.predict_proba(company_input)[0]
                prediction2 = rf_model.predict(company_input)[0]
                
                #SHAP Value
                company_shap_values = explainer.shap_values(company_input)
                
                #Show Prediction Result
                st.markdown("---")
                
                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    if prediction2 == 1:
                        st.success(f"## Prediction: Potable")
                        
                    else:
                        st.error(f"## Prediction: Not Potable")
                        
                
                with result_col2:
                    #Show Confidence Bar
                    st.progress(proba2[1], text=f"Confidence: {proba2[1]*100:.1f}%")
                
                st.markdown("---")
                
                
                #Create 2 function Field
                shap_tab3, shap_tab4 = st.tabs(["Feature Impact", "Dice Advise"])
                
                with shap_tab3:
                    st.markdown("#### Feature Contribution to Potability Prediction (Class1)")
                    st.markdown('<div class="info-box">Show Impact of Each Feature to affect Prediction</div>', unsafe_allow_html=True)
                    
                    fig3, ax3 = plt.subplots(figsize=(12, 8))
                    
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=company_shap_values[0][:, 1],
                            base_values=explainer.expected_value[1],
                            data=company_input.iloc[0].values,
                            feature_names=feature_names
                        ),
                        max_display=15,
                        show=False
                    )
                    plt.title("SHAP Watefall Plot (Local Feature Impact)", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig3)
                
                with shap_tab4:
                
                  if get_dice:
                      st.markdown("### 🔄 DICE Counterfactual Advise")
                      st.markdown('<div class="warning-box">基于反事实解释(CF)的水质处理优化建议，展示如何调整参数以达到水质安全标准。</div>', unsafe_allow_html=True)
                    
                      #DiCE Counterfactual Explanation
                      st.info("Generate Counterfactual Optimization Advise...")
                        
                      from dice_ml import Data, Model, Dice
                        
                      #Create Data Frame for DiCE
                      dice_data = Data(
                            dataframe=pd.concat([X_test, y_test], axis=1),
                            continuous_features=feature_names,
                            outcome_name='Potability'
                      )
                        
                      #Create DiCE Model
                      dice_model = Model(model=rf_model, backend='sklearn')
                        
                      dice_exp = Dice(dice_data, dice_model, method='random')
                        
                      #Dice Explanation
                      if prediction2 == 1:
                            #Potable -> Danger to be polluted
                            desired_class = 0
                            cf_title = "Risk Scenario Simulation: Prevent to Pollution"
                      else:
                            #Not Potable -> Optimization for Water Quality
                            desired_class = 1
                            cf_title = "Optimization Action Proposal"
                        
                      #Counterfactual
                      counterfactuals = dice_exp.generate_counterfactuals(
                            company_input,
                            total_CFs=5,
                            desired_class="opposite"
                      )
                      st.markdown(f"#### {"Input Sample Data"}")
                      st.dataframe(company_input, use_container_width=True)
                      st.markdown("---")
                      st.markdown(f"#### {cf_title}")
                      cf_df = counterfactuals.cf_examples_list[0].final_cfs_df
                      st.dataframe(cf_df, use_container_width=True)
                       
                        
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Attention：The System Advises are intended for reference only!</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== Resident ====================
with tab3:
    st.markdown('<div class="section-header">👨‍👩‍👧‍👦 Resident View - Water Potablity Analysis</div>', unsafe_allow_html=True)
    
    #Split into left/right column
    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.markdown("### 📝 Input Your Water Sample Data")
        st.markdown('<div class="info-box">System Will Analyze Potability Result and Feature Impact</div>', unsafe_allow_html=True)
        
        #User Input
        with st.form("water_quality_form"):
            #Sliders for 9 features
            ph_value = st.slider("**PH**", 0.0, 14.0, 7.0, 0.1 )
            hardness_value = st.slider("**Hardness (mg/L)**", 47.0, 323.0, 150.0, 1.0)
            solids_value = st.slider("**Solids (mg/L)**", 320.0, 61227.0, 20000.0, 100.0)
            chloramines_value = st.slider("**Chloramines (mg/L)**", 0.35, 13.0, 4.0, 0.1)
            sulfate_value = st.slider("**Sulfate (mg/L)**", 129.0, 481.0, 250.0, 1.0)
            conductivity_value = st.slider("**Conductivity (μS/cm)**", 181.0, 753.0, 400.0, 1.0)
            organic_carbon_value = st.slider("**Organic Carbon (mg/L)**", 2.2, 28.0, 10.0, 0.1)
            trihalomethanes_value = st.slider("**Trihalomethanes (μg/L)**", 0.7, 124.0, 50.0, 0.1)
            turbidity_value = st.slider("**Turbidity (NTU)**", 1.45, 6.74, 3.0, 0.1)
            
            #Submission Button
            submitted = st.form_submit_button("🔍 Analyze Water Potablity", type="primary", use_container_width=True)
    
    with col_viz:
        st.markdown("### 📊 Analysis Result")
        
        if submitted:
            #Convert Input to Data Frame
            user_input = pd.DataFrame({
                'ph': [ph_value],
                'Hardness': [hardness_value],
                'Solids': [solids_value],
                'Chloramines': [chloramines_value],
                'Sulfate': [sulfate_value],
                'Conductivity': [conductivity_value],
                'Organic_carbon': [organic_carbon_value],
                'Trihalomethanes': [trihalomethanes_value],
                'Turbidity': [turbidity_value]
            })
            
            #Prediction
            with st.spinner("Analyze Water Quality"):
                
                proba = rf_model.predict_proba(user_input)[0]
                prediction = rf_model.predict(user_input)[0]
                
                #SHAP Value
                user_shap_values = explainer.shap_values(user_input)
                
                #Show Prediction
                st.markdown("---")
                
                if prediction == 1:
                    st.success(f"## Prediction: Potable")
                    
                else:
                    st.error(f"## Prediction: Potable")
                    
                
                #Show Confidence
                st.progress(proba[1], text=f"Confidence: {proba[1]*100:.1f}%")
                
                st.markdown("---")
                
                # SHAP可视化部分
                st.markdown("### Local Feature Impact Analysis")
                
                #Create 2 fields
                shap_tab1, shap_tab2 = st.tabs(["Local Feature Impact", "Decision "])
                
                with shap_tab1:
                    st.markdown("#### 各特征贡献度分析")
                    st.markdown('<div class="info-box">显示每个水质参数对最终预测的具体贡献（正向或负向）</div>', unsafe_allow_html=True)
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    shap.waterfall_plot(
                    shap.Explanation(
                      values=user_shap_values[0,:,1],
                      base_values=explainer.expected_value[1],
                      data=user_input.iloc[0].values,
                      feature_names=feature_names
                    ),
                    max_display=15,
                    show=False
                    )
                    plt.title("SHAP Watefall Plot", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                
                with shap_tab2:
                    st.markdown("#### Decision Visualization")
                    
                    fig_decision, ax_decision = plt.subplots(figsize=(12, 6))
                    shap.decision_plot(
                        explainer.expected_value[1],
                        user_shap_values[:,:,1], 
                        user_input.iloc[0],
                        feature_names=feature_names,
                        feature_order='importance',
                        highlight=0,  
                        show=False
                    )
                    plt.title("SHAP Decision Plot", fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig_decision)
                    
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Attention：System Analysis is base on AI model. It is intended for reference only.</p>
    </div>
    """, unsafe_allow_html=True)
