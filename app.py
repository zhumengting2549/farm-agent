import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ====================== 字体配置【解决中文乱码核心】======================
font_path = "simhei.ttf"
try:
    my_font = fm.FontProperties(fname=font_path)
except Exception:
    my_font = None
plt.rcParams['axes.unicode_minus'] = False

# ====================== 页面基础设置 ======================
st.set_page_config(page_title="水稻玉米种植决策辅助智能体", layout="wide")
st.title("水稻玉米种植决策辅助智能体")
st.markdown("基于可解释AI，辅助小农户做产量、收益种植决策")

# ====================== 模拟样本数据 ======================
np.random.seed(42)
n = 200
data = pd.DataFrame({
    "施肥量": np.random.uniform(20, 80, n),
    "灌溉水量": np.random.uniform(300, 800, n),
    "气温均值": np.random.uniform(18, 32, n),
    "降水": np.random.uniform(50, 250, n),
    "土壤肥力": np.random.uniform(4, 9, n),
})
data["收益"] = (
    data["施肥量"] * 2.2
    + data["灌溉水量"] * 0.12
    + data["气温均值"] * 8
    + data["土壤肥力"] * 25
    - data["降水"] * 0.3
    + np.random.normal(0, 12, n)
)
X = data.drop("收益", axis=1)
y = data["收益"]

# 训练模型
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

# SHAP解释器
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ====================== 侧边栏输入模块 ======================
st.sidebar.header("📝 种植参数输入")
fer = st.sidebar.slider("施肥量", 20.0, 80.0, 45.0)
water = st.sidebar.slider("灌溉水量", 300.0, 800.0, 500.0)
temp = st.sidebar.slider("气温均值", 18.0, 32.0, 24.0)
rain = st.sidebar.slider("降水", 50.0, 250.0, 120.0)
soil = st.sidebar.slider("土壤肥力", 4.0, 9.0, 6.5)

input_df = pd.DataFrame([[fer, water, temp, rain, soil]], columns=X.columns)
pred = model.predict(input_df)[0]
st.subheader("📊 预测结果")
st.metric("预估收益(元/亩)", f"{pred:.2f}")

# ====================== 【修复版本】SHAP因子重要性柱状图，不再用shap自带bar图 ======================
st.subheader("图2：SHAP因子影响分析（可解释AI核心）")
# 计算每个特征平均SHAP绝对值（重要性）
shap_importance = np.abs(shap_values).mean(axis=0)
feature_names = X.columns

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(feature_names, shap_importance)

if my_font is not None:
    ax.set_title("SHAP因子影响分析（可解释AI核心）", fontproperties=my_font, fontsize=14)
    ax.set_xlabel("SHAP重要性分值", fontproperties=my_font)
    ax.set_yticklabels(feature_names, fontproperties=my_font)
ax.invert_yaxis()
st.pyplot(fig)

# ====================== SHAP瀑布图（单样本解释，这个相对稳定） ======================
st.subheader("单样本因子贡献瀑布图")
try:
    fig2, ax2 = plt.subplots(figsize=(9,5))
    shap.waterfall_plot(explainer(input_df)[0], show=False)
    if my_font is not None:
        ax2.set_title("单条样本各因子对收益的贡献", fontproperties=my_font)
    st.pyplot(fig2)
except Exception as e:
    st.warning("瀑布图渲染受限，不影响预测与重要性分析")

st.markdown("---")
st.caption("第二届全国人工智能应用创新大赛｜小农户产量收益双目标可解释助农Agent")