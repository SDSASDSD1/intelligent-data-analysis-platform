"""工作台的展示样式与示例问题；不负责数据计算或权限判断。"""

import streamlit as st


PRIMARY = "#2563EB"
SUMMARY_QUESTION = "统计销售额总和和平均值。"
WORKFLOW_QUESTION = "先查看字段，再统计销售额，最后选择柱状图。"

STYLES = """
<style>
.stApp {
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}
.stMainBlockContainer {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.st-key-workspace_header {
    background: linear-gradient(115deg, #eaf1ff 0%, #f5f9ff 65%, #e4efff 100%);
    border: 1px solid #d9e6fb;
    border-radius: 22px;
    padding: 1.7rem 2rem;
    margin-bottom: 1.2rem;
}
.st-key-workspace_header h1 {
    font-size: clamp(1.7rem, 3.2vw, 2.5rem);
    letter-spacing: -.04em;
    padding-top: .25rem;
    padding-bottom: .3rem;
}
.workspace-eyebrow {
    color: #2563eb;
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .14em;
}
.workspace-tags { display: flex; flex-wrap: wrap; gap: .5rem; }
.workspace-tags span {
    display: inline-block;
    padding: .35rem .75rem;
    border: 1px solid #d3e1f8;
    border-radius: 999px;
    color: #315786;
    background: #ffffffb8;
    font-size: .8rem;
}
[class*="st-key-panel_"] {
    background: #fff;
    border: 1px solid #e2e9f3;
    border-radius: 16px;
    padding: 1.25rem;
    box-shadow: 0 4px 16px #15355c05;
}
.st-key-panel_upload { border: 1px dashed #9ebcf2; }
.st-key-panel_agent { border-top: 4px solid #2563eb; }
.st-key-panel_review { border-left: 4px solid #d99419; background: #fffcf5; }
.st-key-panel_result { border-left: 4px solid #15957a; }
.st-key-panel_login { max-width: 480px; margin: 1rem auto; padding: 2rem; }
.st-key-panel_account { padding: 1rem; }
.st-key-workspace_metrics [data-testid="stMetric"] {
    background: #fff;
    border: 1px solid #e2e9f3;
    border-top: 3px solid #719bf0;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    min-height: 116px;
}
.st-key-workspace_metrics [data-testid="stMetricValue"] {
    color: #1e4ca4;
    font-size: clamp(1.4rem, 2.2vw, 2rem);
}
.workspace-steps { display: flex; flex-wrap: wrap; gap: .75rem; margin: .25rem 0 1rem; }
.workspace-step { color: #5c6f89; font-size: .85rem; }
.workspace-step b {
    display: inline-flex; justify-content: center; align-items: center;
    width: 24px; height: 24px; border-radius: 8px; background: #eaf1ff;
    color: #2563eb; margin-right: .4rem;
}
.workspace-feature-icon {
    width: 40px; height: 40px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    color: #2563eb; background: #eaf1ff; font-size: 1.3rem;
}
.st-key-panel_agent button, .st-key-panel_login button {
    border-radius: 10px;
}
@media (max-width: 768px) {
    .stMainBlockContainer { padding: 1rem; }
    .st-key-workspace_header { padding: 1.2rem; border-radius: 16px; }
    [class*="st-key-panel_"] { padding: 1rem; }
    .st-key-workspace_metrics [data-testid="stMetric"] { padding: .85rem; }
    .st-key-workspace_metrics [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    .st-key-workspace_metrics [data-testid="stColumn"] {
        flex: 1 1 calc(50% - 1rem);
        min-width: calc(50% - 1rem);
    }
}
@media (max-width: 480px) {
    .st-key-workspace_metrics [data-testid="stColumn"] {
        flex-basis: 100%;
        min-width: 100%;
    }
}
</style>
"""


def configure_page():
    st.set_page_config(
        page_title="智能数据分析平台", page_icon="📊",
        layout="wide", initial_sidebar_state="expanded",
    )
    # 此 HTML 只包含项目内固定样式；用户内容始终交给原生文本组件。
    st.html(STYLES)


def render_header():
    with st.container(key="workspace_header"):
        st.html('<div class="workspace-eyebrow">DATA WORKSPACE · 数据分析工作台</div>')
        st.title("智能数据分析平台")
        st.write("从一份表格开始，让数据成为清晰的答案。")
        st.html(
            '<div class="workspace-tags"><span>▦ CSV / Excel</span>'
            '<span>✦ 智能分析</span><span>✓ 人工审核</span></div>'
        )


def render_steps():
    st.html(
        '<div class="workspace-steps">'
        '<span class="workspace-step"><b>1</b>上传数据</span>'
        '<span class="workspace-step"><b>2</b>核对字段</span>'
        '<span class="workspace-step"><b>3</b>开始分析</span></div>'
    )


def render_empty_state():
    st.subheader("开始你的第一次分析")
    st.caption("上传销售数据后，即可查看统计结果、生成图表，并向 Agent 提问。")
    features = [
        ("▦", "看清数据", "预览前 10 行，检查字段、缺失值与销售指标。"),
        ("✦", "用自然语言分析", "描述你的需求，由受控工具完成统计与图表选择。"),
        ("✓", "记录每一步", "敏感操作先审核，分析完成后查看过程并下载报告。"),
    ]
    for index, (column, feature) in enumerate(zip(st.columns(3), features)):
        symbol, title, description = feature
        with column, st.container(key=f"panel_feature_{index}"):
            st.html(f'<div class="workspace-feature-icon">{symbol}</div>')
            st.markdown(f"**{title}**")
            st.caption(description)


def fill_agent_question(question):
    """示例按钮仅填入输入框；模型调用仍由“开始 Agent 分析”触发。"""
    st.session_state["agent_question"] = question


def render_question_examples():
    st.caption("试试这样提问 · 点击填入后，再开始分析")
    with st.container(horizontal=True, wrap=True):
        st.button(
            "汇总销售指标", key="agent_example_summary",
            on_click=fill_agent_question, args=(SUMMARY_QUESTION,),
        )
        st.button(
            "查字段 → 汇总 → 柱状图", key="agent_example_workflow",
            on_click=fill_agent_question, args=(WORKFLOW_QUESTION,),
        )


def style_chart(fig, ax):
    """统一图表外观；坐标、数据和中文字体由原绘图代码控制。"""
    fig.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5ECF6", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D8E2EF")
    ax.tick_params(colors="#586B85", length=0)
    ax.xaxis.label.set_color("#586B85")
    ax.yaxis.label.set_color("#586B85")
    ax.title.set_color("#172B4D")
    ax.title.set_fontsize(12)
    ax.title.set_fontweight("bold")

