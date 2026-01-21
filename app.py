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

# 设置页面配置
st.set_page_config(
    page_title="水质安全XAI解释系统",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
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

# 应用主标题
st.markdown('<div class="main-title">💧 水质安全预测模型可解释性(XAI)系统</div>', unsafe_allow_html=True)

# 角色选择 - 使用水平选项卡
st.markdown("### 请选择您的角色视角：")
tab1, tab2, tab3 = st.tabs(["📊 **水质监管部门**", "🏭 **供水公司**", "👨‍👩‍👧‍👦 **居民**"])

# 侧边栏信息
with st.sidebar:
    st.write("💧")
    st.markdown("### 系统信息")
    st.markdown("""
    **版本**: 1.0.0  
    **模型**: Random Forest  
    **XAI方法**: SHAP  
    **数据**: Water Potability Dataset
    """)
    
    st.markdown("---")
    st.markdown("### 使用说明")
    st.markdown("""
    1. 选择上方的角色标签
    2. 查看对应角色的XAI解释
    3. 展开/收起特征分析部分
    4. 所有分析基于同一模型
    """)
    
    st.markdown("---")
    st.markdown("### 特征说明")
    st.markdown("""
    - **ph**: 酸碱度 (0-14)
    - **Hardness**: 硬度 (mg/L)
    - **Solids**: 总溶解固体 (mg/L)
    - **Chloramines**: 氯胺 (mg/L)
    - **Sulfate**: 硫酸盐 (mg/L)
    - **Conductivity**: 电导率 (μS/cm)
    - **Organic_carbon**: 有机碳 (mg/L)
    - **Trihalomethanes**: 三卤甲烷 (μg/L)
    - **Turbidity**: 浊度 (NTU)
    """)

# ==================== 数据加载和模型训练函数 ====================
@st.cache_resource
def load_data_and_train():
    """加载数据并训练模型，返回所有必要对象"""
    
    # 显示加载进度
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 步骤1: 加载数据
    status_text.text("步骤1/4: 加载数据集...")
    progress_bar.progress(25)
    
    try:
        # 从data文件夹读取数据
        df = pd.read_csv("data/water_potability.csv")
    except FileNotFoundError:
        st.error("找不到数据文件: data/water_potability.csv")
        st.stop()
    
    # 步骤2: 数据处理
    status_text.text("步骤2/4: 处理数据...")
    progress_bar.progress(50)
    
    # 处理缺失值
    for col in ['ph', 'Sulfate', 'Trihalomethanes']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    
    # 准备特征和目标
    feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    X = df[feature_names].copy()
    y = df['Potability'].copy()
    
    # 步骤3: 训练模型
    status_text.text("步骤3/4: 训练模型...")
    progress_bar.progress(75)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 处理类别不平衡
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    # 训练随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    # 计算模型性能
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # 步骤4: 计算SHAP值
    status_text.text("步骤4/4: 计算SHAP解释值...")
    progress_bar.progress(95)
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(rf_model)
    
    # 计算SHAP值（只计算前200个样本来加速）
    shap_values = explainer.shap_values(X_test)
    
    # 计算SHAP值的标准差
    shap_std = {}
    if shap_values is not None and len(shap_values) > 1:
        shap_class1 = shap_values[:,:,1]  # 类别1的SHAP值
        for i, feature in enumerate(feature_names):
            shap_std[feature] = np.std(shap_class1[:, i])
    
    status_text.text("准备完成!")
    progress_bar.progress(100)
    
    # 清理进度显示
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

# ==================== 主应用逻辑 ====================

# 加载数据并训练模型（带缓存）
with st.spinner('正在初始化模型和计算SHAP值...'):
    data_dict = load_data_and_train()

# 提取数据
rf_model = data_dict['model']
explainer = data_dict['explainer']
shap_values = data_dict['shap_values']
X_test = data_dict['X_test']
feature_names = data_dict['feature_names']
shap_std = data_dict['shap_std']
metrics = data_dict['metrics']

# ==================== 监管部门界面 ====================
with tab1:
    st.markdown('<div class="section-header">📊 水质监管部门 - XAI分析面板</div>', unsafe_allow_html=True)
    
    # 显示模型性能
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("准确率", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("精确率", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("召回率", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("F1分数", f"{metrics['f1']:.3f}")
    
    st.markdown("---")
    
    # 第一部分: SHAP摘要图
    st.markdown('<div class="section-header">1. 全局特征重要性分析</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">此图显示各特征对"水质安全"(类别1)预测的平均影响程度，帮助识别关键监管指标。</div>', unsafe_allow_html=True)
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    shap.summary_plot(
        shap_values[:,:,1],  # 类别1的SHAP值
        X_test,
        feature_names=feature_names,
        show=False,
        plot_type="dot"
    )
    plt.title("SHAP特征重要性摘要图 (类别1: 水质安全)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1)
    
    # 第二部分: SHAP值标准差分布
    st.markdown('<div class="section-header">2. 风险波动分析 - SHAP值标准差分布</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box">标准差越大，表示该特征在不同样本中的影响波动越大，监管时需要特别关注其变化范围。</div>', unsafe_allow_html=True)
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    features = list(shap_std.keys())
    std_values = list(shap_std.values())
    
    # 排序以便更好地展示
    sorted_indices = np.argsort(std_values)[::-1]
    features_sorted = [features[i] for i in sorted_indices]
    values_sorted = [std_values[i] for i in sorted_indices]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features_sorted)))
    bars = ax2.barh(features_sorted, values_sorted, color=colors)
    
    # 添加数值标签
    for bar, value in zip(bars, values_sorted):
        width = bar.get_width()
        ax2.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left', va='center')
    
    ax2.set_xlabel('SHAP值标准差', fontsize=12)
    ax2.set_title('各特征SHAP值波动程度 (类别1)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig2)
    
    # 第三部分: 特征依赖关系分析 (可折叠分栏)
    st.markdown('<div class="section-header">3. 详细特征依赖关系分析</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">点击展开查看每个特征如何影响水质安全预测，了解特征的边际效应。</div>', unsafe_allow_html=True)
    
    # 添加全部展开/收起按钮
    col_expand1, col_expand2 = st.columns(2)
    with col_expand1:
        if st.button("🔼 全部收起", use_container_width=True):
            st.session_state.expand_all = False
    with col_expand2:
        if st.button("🔽 全部展开", use_container_width=True):
            st.session_state.expand_all = True
    
    # 初始化session state
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
    
    # 为每个特征创建可折叠分栏
    for i, feature in enumerate(feature_names):
        with st.expander(f"**{feature}** - {feature_descriptions.get(feature, '')}", 
                        expanded=st.session_state.expand_all):
            
            st.markdown(f'<div class="feature-card">', unsafe_allow_html=True)
            
            # 创建两列布局
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                # 生成依赖图
                fig_dep, ax_dep = plt.subplots(figsize=(8, 4))
                shap.dependence_plot(
                    feature,
                    shap_values[:,:,1],
                    X_test,
                    feature_names=feature_names,
                    ax=ax_dep,
                    show=False
                )
                ax_dep.set_title(f'{feature} SHAP依赖图', fontsize=12, fontweight='bold')
                ax_dep.set_xlabel(feature, fontsize=10)
                ax_dep.set_ylabel('SHAP值 (对类别1的影响)', fontsize=10)
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

# ==================== 供水公司界面 (预留) ====================
with tab2:
    st.markdown('<div class="section-header">🏭 供水公司视角 - 水质处理优化</div>', unsafe_allow_html=True)
    
    # 供水公司页面布局分为两列
    col_input2, col_viz2 = st.columns([1, 2])
    
    with col_input2:
        st.markdown("### 📝 输入水质处理参数")
        st.markdown('<div class="info-box">请输入当前处理工艺的水质参数，系统将提供优化建议。</div>', unsafe_allow_html=True)
        
        # 使用session_state保存输入值
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
        
        # 用户输入表单
        with st.form("company_water_quality_form"):
            # 创建9个特征输入框 - 使用session_state中的值
            ph_value2 = st.slider("**ph值**", 0.0, 14.0, st.session_state.company_inputs['ph'], 0.1,
                                help="建议处理范围: 6.8-8.2", key='ph2')
            hardness_value2 = st.slider("**硬度 (mg/L)**", 47.0, 323.0, st.session_state.company_inputs['Hardness'], 1.0,
                                       help="软化处理建议值: 80-120", key='hardness2')
            solids_value2 = st.slider("**总溶解固体 (mg/L)**", 320.0, 61227.0, st.session_state.company_inputs['Solids'], 100.0,
                                    help="反渗透处理目标: < 500", key='solids2')
            chloramines_value2 = st.slider("**氯胺 (mg/L)**", 0.35, 13.0, st.session_state.company_inputs['Chloramines'], 0.1,
                                         help="消毒副产物控制: < 3", key='chloramines2')
            sulfate_value2 = st.slider("**硫酸盐 (mg/L)**", 129.0, 481.0, st.session_state.company_inputs['Sulfate'], 1.0,
                                     help="离子交换处理目标: 200-300", key='sulfate2')
            conductivity_value2 = st.slider("**电导率 (μS/cm)**", 181.0, 753.0, st.session_state.company_inputs['Conductivity'], 1.0,
                                          help="反映处理效率: 300-500", key='conductivity2')
            organic_carbon_value2 = st.slider("**有机碳 (mg/L)**", 2.2, 28.0, st.session_state.company_inputs['Organic_carbon'], 0.1,
                                            help="活性炭过滤目标: < 5", key='organic_carbon2')
            trihalomethanes_value2 = st.slider("**三卤甲烷 (μg/L)**", 0.7, 124.0, st.session_state.company_inputs['Trihalomethanes'], 0.1,
                                             help="关键控制指标: < 60", key='trihalomethanes2')
            turbidity_value2 = st.slider("**浊度 (NTU)**", 1.45, 6.74, st.session_state.company_inputs['Turbidity'], 0.1,
                                       help="絮凝沉淀目标: < 1", key='turbidity2')
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                # 提交按钮
                submitted2 = st.form_submit_button("🔍 分析水质安全性", type="primary", use_container_width=True)
            with col_btn2:
                # 获取Dice建议按钮
                get_dice = st.form_submit_button("🔄 获取优化方案", type="secondary", use_container_width=True)
    
    with col_viz2:
        st.markdown("### 📊 分析结果与优化建议")
        
        # 检查是否有提交
        if submitted2 or get_dice:
            # 更新session_state中的值
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
            
            # 创建输入数据的DataFrame
            company_input = pd.DataFrame([st.session_state.company_inputs])
            
            # 进行预测
            with st.spinner("正在分析水质并计算建议..."):
                # 预测概率和类别
                proba2 = rf_model.predict_proba(company_input)[0]
                prediction2 = rf_model.predict(company_input)[0]
                
                # 计算SHAP值
                company_shap_values = explainer.shap_values(company_input)
                
                # 显示预测结果
                st.markdown("---")
                
                # 创建结果卡片
                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    if prediction2 == 1:
                        st.success(f"## ✅ 出水水质安全")
                        st.metric("达标概率", f"{proba2[1]*100:.1f}%", delta="达标", delta_color="normal")
                    else:
                        st.error(f"## ⚠️ 出水水质不达标")
                        st.metric("不达标概率", f"{proba2[0]*100:.1f}%", delta="需改进", delta_color="inverse")
                
                with result_col2:
                    # 显示置信度条
                    st.progress(proba2[1], text=f"安全置信度: {proba2[1]*100:.1f}%")
                    st.caption(f"处理效果评分: {proba2[1]*100:.0f}/100")
                
                st.markdown("---")
                
                
                # 创建两个选项卡
                shap_tab3, shap_tab4 = st.tabs(["参数贡献度", "决策路径"])
                
                with shap_tab3:
                    st.markdown("#### 各工艺参数贡献度分析")
                    st.markdown('<div class="info-box">显示每个处理参数对最终出水水质的贡献度，识别关键控制点。</div>', unsafe_allow_html=True)
                    
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
                    plt.title("工艺参数贡献度瀑布图（供水公司视角）", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig3)
                
                with shap_tab4:
                    # 如果点击了"获取优化方案"，显示Dice建议
                  if get_dice:
                      st.markdown("### 🔄 DICE优化方案")
                      st.markdown('<div class="warning-box">基于反事实解释(CF)的水质处理优化建议，展示如何调整参数以达到水质安全标准。</div>', unsafe_allow_html=True)
                    
                      # 使用Dice生成反事实解释
                      st.info("正在生成反事实优化方案...")
                        
                      # 创建Dice解释器
                      from dice_ml import Data, Model, Dice
                        
                      # 准备Dice所需的数据对象
                      dice_data = Data(
                            dataframe=pd.concat([X_test, y_test], axis=1),
                            continuous_features=feature_names,
                            outcome_name='Potability'
                      )
                        
                      # 创建Dice模型对象
                      dice_model = Model(model=rf_model, backend='sklearn')
                        
                      # 创建Dice解释器
                      dice_exp = Dice(dice_data, dice_model, method='random')
                        
                      # 生成反事实解释
                      if prediction2 == 1:
                            # 如果当前是安全的，生成如何可能变得不安全的反事实
                            desired_class = 0
                            cf_title = "风险情景模拟：可能导致水质不达标的参数变化"
                      else:
                            # 如果当前是不安全的，生成如何变得安全的反事实
                            desired_class = 1
                            cf_title = "优化建议：实现水质达标的参数调整方案"
                        
                      # 生成反事实
                     counterfactuals = dice_exp.generate_counterfactuals(
                            company_input,
                            total_CFs=5,
                            desired_class=desired_class,
                            proximity_weight=1.0,
                            diversity_weight=1.0,
                            features_to_vary='all'
                      )
                        
                        
    
    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🏭 注意：本系统建议仅供参考，实际工艺调整需结合现场条件和专家意见。</p>
    <p>DICE算法基于机器学习模型生成反事实解释，建议进行小试验证。</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== 居民界面 (预留) ====================
with tab3:
    st.markdown('<div class="section-header">👨‍👩‍👧‍👦 居民用户视角 - 我家水质分析</div>', unsafe_allow_html=True)
    
    # 居民页面布局分为两列
    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.markdown("### 📝 输入水质参数")
        st.markdown('<div class="info-box">请输入您家的水质检测数据，系统将分析安全性和影响因素。</div>', unsafe_allow_html=True)
        
        # 用户输入表单
        with st.form("water_quality_form"):
            # 创建9个特征输入框
            ph_value = st.slider("**ph值 (酸碱度)**", 0.0, 14.0, 7.0, 0.1, 
                                help="0-14范围，7为中性，6.5-8.5为安全范围")
            hardness_value = st.slider("**Hardness (硬度 mg/L)**", 47.0, 323.0, 150.0, 1.0,
                                      help="47-323 mg/L，适中硬度对健康有益")
            solids_value = st.slider("**Solids (总溶解固体 mg/L)**", 320.0, 61227.0, 20000.0, 100.0,
                                   help="320-61227 mg/L，反映水中矿物质含量")
            chloramines_value = st.slider("**Chloramines (氯胺 mg/L)**", 0.35, 13.0, 4.0, 0.1,
                                        help="0.35-13 mg/L，消毒副产物，应低于4 mg/L")
            sulfate_value = st.slider("**Sulfate (硫酸盐 mg/L)**", 129.0, 481.0, 250.0, 1.0,
                                    help="129-481 mg/L，过高可能引起不适")
            conductivity_value = st.slider("**Conductivity (电导率 μS/cm)**", 181.0, 753.0, 400.0, 1.0,
                                         help="181-753 μS/cm，反映离子总量")
            organic_carbon_value = st.slider("**Organic Carbon (有机碳 mg/L)**", 2.2, 28.0, 10.0, 0.1,
                                           help="2.2-28 mg/L，微生物营养源")
            trihalomethanes_value = st.slider("**Trihalomethanes (三卤甲烷 μg/L)**", 0.7, 124.0, 50.0, 0.1,
                                            help="0.7-124 μg/L，潜在致癌物，应低于80 μg/L")
            turbidity_value = st.slider("**Turbidity (浊度 NTU)**", 1.45, 6.74, 3.0, 0.1,
                                      help="1.45-6.74 NTU，越低越清澈")
            
            # 提交按钮
            submitted = st.form_submit_button("🔍 分析我家水质", type="primary", use_container_width=True)
    
    with col_viz:
        st.markdown("### 📊 分析结果")
        
        if submitted:
            # 创建输入数据的DataFrame
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
            
            # 进行预测
            with st.spinner("正在分析水质..."):
                # 预测概率和类别
                proba = rf_model.predict_proba(user_input)[0]
                prediction = rf_model.predict(user_input)[0]
                
                # 计算SHAP值
                user_shap_values = explainer.shap_values(user_input)
                
                # 显示预测结果
                st.markdown("---")
                
                # 创建结果卡片
                if prediction == 1:
                    st.success(f"## ✅ 水质安全可饮用")
                    st.metric("安全概率", f"{proba[1]*100:.1f}%", delta="安全", delta_color="normal")
                else:
                    st.error(f"## ⚠️ 水质不推荐饮用")
                    st.metric("不安全概率", f"{proba[0]*100:.1f}%", delta="风险", delta_color="inverse")
                
                # 显示置信度条
                st.progress(proba[1], text=f"可饮用置信度: {proba[1]*100:.1f}%")
                
                st.markdown("---")
                
                # SHAP可视化部分
                st.markdown("### 🔬 影响因素分析")
                
                # 创建两个选项卡：力图和决策图
                shap_tab1, shap_tab2 = st.tabs(["单个特征影响", "决策过程追踪"])
                
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
                    plt.title("特征贡献度瀑布图", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                
                with shap_tab2:
                    st.markdown("#### 决策过程可视化")
                    
                    # 创建决策图
                    fig_decision, ax_decision = plt.subplots(figsize=(12, 6))
                    shap.decision_plot(
                        explainer.expected_value[1],
                        user_shap_values[:,:,1], 
                        user_input.iloc[0],
                        feature_names=feature_names,
                        feature_order='importance',
                        highlight=0,  # 高亮显示用户输入（第一个）
                        show=False
                    )
                    plt.title("决策路径分析", fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig_decision)
                    
    
    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>💧 注意：本分析基于机器学习模型预测，仅供参考。如有健康疑虑，请咨询专业机构。</p>
    <p>数据更新频率：模型每月更新 | 最后更新：本月</p>
    </div>
    """, unsafe_allow_html=True)
# ==================== 页脚 ====================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>💧 水质安全XAI解释系统 | 基于SHAP的模型可解释性分析</p>
        <p>数据来源: Water Potability Dataset | 模型: Random Forest Classifier</p>
        <p>© 2024 水质监管科技平台 | 版本 1.0.0</p>
    </div>
    """,
    unsafe_allow_html=True
)
