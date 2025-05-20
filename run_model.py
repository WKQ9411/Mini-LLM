# -*- coding: utf-8 -*-
import streamlit as st
import torch
from transformers import AutoTokenizer
import time
import traceback  # 用于更详细的错误追踪
import html
from model import get_model_and_args
from utils.little_tools import load_yaml
import inspect


# --- 定义分词器 ---
TOKENIZER_NAME = "./mini_tokenizer"

# --- 模型配置定义，根据模型权重实际路径修改 ---
MODEL_CONFIG_DEFINITIONS = {
    "mini_deepseekv3": {
        "path": "C:/Users/WKQ/Downloads/sft_model/mini_deepseekv3/sft_mini_deepseekv3_final_loss_2.09-chat.pt",
        "yaml": "C:/Users/WKQ/Downloads/sft_model/mini_deepseekv3/sft_mini_deepseekv3_model_args.yaml",
        "inference_args": {"use_noaux_tc": True},  # 定义模型特有推理参数
        "description": "MoE",
    },
    "mini_llama3": {
        "path": "C:/Users/WKQ/Downloads/sft_model/mini_llama3/sft_mini_llama3_final_loss_2.52-chat.pt",
        "yaml": "C:/Users/WKQ/Downloads/sft_model/mini_llama3/sft_mini_llama3_model_args.yaml",
        "inference_args": {},
        "description": "Dense",
    },
}


# --- 函数定义 ---
# 加载模型结构
@st.cache_resource(show_spinner="初始化模型结构...")
def get_cached_model_structure(model_name, config_dict):
    try:
        Model, Model_Args = get_model_and_args(model_name)
        model_args = Model_Args(**config_dict)
        model = Model(model_args)
        model.eval()
        return model, model_args
    except Exception as e: st.error(f"初始化模型结构时出错 ({model_name}): {e}"); return None, None

# 加载模型权重
@st.cache_resource(show_spinner="加载模型权重...")
def load_cached_weights(_model, model_path, device):
    if _model is None: 
        return None, "模型结构未成功初始化"
    try: 
        _model.load_state_dict(torch.load(model_path, map_location=device))
        _model.to(device)
        return _model, "success"
    except FileNotFoundError: 
        return None, f"模型文件未找到: {model_path}"
    except Exception as e: 
        return None, f"加载权重时出错: {str(e)}\n{traceback.format_exc()}"

# 加载分词器
@st.cache_resource(show_spinner="加载分词器...")
def load_cached_tokenizer(tokenizer_name):
    try: 
        return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True), "success"
    except Exception as e: 
        return None, f"加载分词器时出错 '{tokenizer_name}': {e}\n{traceback.format_exc()}"


# --- 页面配置 ---
st.set_page_config( page_title="Mini-LLM", page_icon="./utils/Logo.svg", layout="wide", initial_sidebar_state="expanded", )

# --- 自定义CSS样式 ---
st.markdown(
    """
<style>
    /* 全局设置 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* 主题渐变色调 */
    :root {
        --primary-gradient-start: #0a192f;
        --primary-gradient-end: #172a45;
        --accent-color: #64ffda;
        --card-bg: rgba(255, 255, 255, 0.08);
        --card-border: rgba(255, 255, 255, 0.1);
    }

    /* 修复白色头部部分，使其与主题匹配 */
    header {
        background: linear-gradient(135deg, var(--primary-gradient-start) 0%, var(--primary-gradient-end) 100%) !important;
        color: white !important;
    }

    /* 主应用样式与匹配的渐变 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        background: linear-gradient(135deg, var(--primary-gradient-start) 0%, var(--primary-gradient-end) 100%);
        color: white;
        min-height: 100vh;
    }

    /* 侧边栏样式，匹配渐变色 */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(170deg, var(--primary-gradient-start) 0%, var(--primary-gradient-end) 100%);
        color: white;
        padding-top: 1rem;
        box-shadow: 2px 0 20px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stSidebar"] h1 {
        color: white;
        font-weight: 600;
        font-size: 1.6em;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2);
        margin: 15px 0;
    }

    /* 玻璃拟态卡片 */
    .sidebar-section-header {
        font-size: 1.1em;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 15px;
        padding: 10px 12px;
        color: #f0f0f0;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        display: flex;
        align-items: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    .sidebar-section-header span {
        margin-right: 8px;
        font-size: 1.2em;
    }

    /* 发光按钮样式 */
    [data-testid="stSidebar"] button {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        text-transform: uppercase;
        font-size: 0.85em;
        letter-spacing: 0.5px;
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }

    [data-testid="stSidebar"] button:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: var(--accent-color);
        box-shadow: 0 0 15px rgba(100, 255, 218, 0.6);
        transform: translateY(-1px);
    }

    /* 玻璃卡片样式 */
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: white;
    }
    
    /* 用户消息容器 */
    .message-container.user {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 8px;
        margin-right: 8px;
    }
    
    /* 用户气泡样式 */
    .user-bubble {
        max-width: 80%;
        margin-left: auto;
    }
    
    /* 用户气泡样式 */
    .user-bubble {
        background: rgba(100, 255, 218, 0.15);
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        max-width: 80%;
        margin-left: auto;
        border: 1px solid rgba(100, 255, 218, 0.3);
    }

    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.15), 0 0 15px rgba(100, 255, 218, 0.3);
    }

    .info-card h4 {
        margin-top: 0;
        margin-bottom: 15px;
        color: white;
        font-size: 1.1em;
        font-weight: 700;
        display: flex;
        align-items: center;
        border-bottom: 2px solid rgba(255, 255, 255, 0.2);
        padding-bottom: 10px;
    }

    .info-card h4 span {
        margin-right: 10px;
        font-size: 1.2em;
        color: var(--accent-color);
    }

    .info-card p {
        margin-bottom: 10px;
        line-height: 1.7;
        color: rgba(255, 255, 255, 0.9);
        display: flex;
        align-items: center;
        font-size: 0.95em;
        padding: 3px 0;
    }

    .info-card p span {
        margin-right: 10px;
        width: 22px;
        text-align: center;
        color: var(--accent-color);
    }

    .info-card strong {
        color: white;
        margin-right: 5px;
        font-weight: 600;
    }

    /* 自定义加载动画文本样式 */
    .stSpinner > div > div {
       color: var(--accent-color);
       font-weight: 500;
    }

    /* 重新定义聊天布局 */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1.5rem; /* Reduced gap for better density */
        padding: 1rem;
        max-width: 100%;
    }

    /* 消息容器样式 */
    .message-container {
        display: flex;
        flex-direction: column;
        max-width: 85%;
        margin-left: auto;
        margin-right: auto;
        transition: transform 0.2s ease;
    }

    .message-container.assistant {
        align-items: flex-start;
        align-self: flex-start;
        margin-left: 0;
        margin-right: auto;
        margin-bottom: 16px;
    }

    /* 头像样式 */
    .avatar {
        width: 38px; height: 38px; border-radius: 50%; display: flex;
        align-items: center; justify-content: center; font-size: 16px;
        margin-bottom: 6px; box-shadow: 0 3px 6px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
    }
    .avatar:hover { transform: scale(1.1) rotate(5deg); }
    .avatar.user { background: linear-gradient(135deg, #4338ca, #3b82f6); color: white; margin-left: auto; }
    .avatar.assistant { background: linear-gradient(135deg, #6d28d9, #8b5cf6); color: white; margin-right: auto; }

    /* 聊天气泡样式 - 宽度适配 */
    .chat-bubble {
        padding: 0.8rem 1rem; border-radius: 1rem; position: relative;
        margin-bottom: 0.3rem; animation: fadeIn 0.3s ease;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
        width: fit-content; /* --- 适配内容宽度 --- */
        max-width: 100%; word-wrap: break-word; line-height: 1.6;
    }
    .chat-bubble span { display: block; } /* 允许内容块级显示和换行 */
    @keyframes fadeIn { from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: translateY(0);} }
    .user-bubble { background: linear-gradient(135deg, #4338ca, #3b82f6); color: white; border-top-right-radius: 0.25rem; box-shadow: 0 3px 10px rgba(59, 130, 246, 0.3); }
    .assistant-bubble { 
        background: linear-gradient(135deg, #1e293b, #334155); 
        color: #f3f4f6; 
        border-top-left-radius: 0.25rem; 
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.3);
        margin-left: 0;
        transform-origin: left center;
    }

    /* 气泡信息栏样式 */
    .bubble-info { font-size: 0.7rem; opacity: 1; display: flex; justify-content: flex-start; gap: 10px; padding-top: 0.4rem; margin-top: 0.4rem; border-top: 1px solid rgba(255, 255, 255, 0.15); animation: fadeIn 0.3s ease forwards; }
    .bubble-info span { display: inline-flex; align-items: center; }
    .bubble-info span svg, .bubble-info span i { margin-right: 3px; font-size: 0.8em; }

    /* 标题样式 */
    .main h1 { color: #ffffff; font-weight: 800; letter-spacing: -0.5px; padding-bottom: 10px; border-bottom: 2px solid rgba(255, 255, 255, 0.2); margin-bottom: 20px; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); }

    /* 说明文本样式 */
    .main .caption-text { font-style: italic; color: rgba(255, 255, 255, 0.9); margin-bottom: 20px; padding: 12px 18px; background-color: rgba(255, 255, 255, 0.1); backdrop-filter: blur(5px); -webkit-backdrop-filter: blur(5px); border-radius: 8px; border-left: 3px solid var(--accent-color); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }

    /* 玻璃拟态设置面板 */
    .stExpander { border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.15); margin-top: 10px; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); overflow: hidden; }
    .stExpander > div:first-child { padding: 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 10px 10px 0 0; }
    .stExpander > div:last-child { border-top: 1px solid rgba(255, 255, 255, 0.1); padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 0 0 10px 10px; }
    /* 滑块控件样式 */
    .stSlider > div > div > div { background-color: rgba(255, 255, 255, 0.2) !important; }
    .stSlider > div > div > div > div { background-color: var(--accent-color) !important; }
    /* 移动设备响应式设计 */
    @media (max-width: 768px) { .main .block-container { padding: 1rem 0.5rem; } .message-container { max-width: 95%; } .info-card { padding: 12px 15px; } .main h1 { font-size: 1.8em; } .chat-container { gap: 1rem; } }
    /* 现代动画进度条 */
    .progress-bar-container { width: 100%; height: 6px; background-color: rgba(255, 255, 255, 0.1); border-radius: 3px; margin: 10px 0; overflow: hidden; }
    .progress-bar { height: 100%; background: linear-gradient(90deg, var(--primary-gradient-start) 0%, var(--accent-color) 100%); background-size: 200% 100%; animation: loading-indeterminate 1.5s linear infinite; border-radius: 3px; }
    @keyframes loading-indeterminate { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
    /* 加载元素的脉冲动画 */
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    .pulse { animation: pulse 1.5s infinite; }
    /* 状态提示框样式 */
    .alert-box { padding: 12px 16px; border-radius: 10px; margin: 15px 0; display: flex; align-items: center; backdrop-filter: blur(5px); -webkit-backdrop-filter: blur(5px); }
    .alert-box.error { background-color: rgba(239, 68, 68, 0.2); border-left: 4px solid #ef4444; }
    .alert-box.info { background-color: rgba(100, 255, 218, 0.15); border-left: 4px solid var(--accent-color); }
    .alert-box.warning { background-color: rgba(245, 158, 11, 0.15); border-left: 4px solid #f59e0b; }
    /* 参数指示器样式 */
    .param-indicator { display: flex; justify-content: space-between; font-size: 0.8em; color: rgba(255, 255, 255, 0.8); margin-bottom: 5px; }
    .param-meter { width: 100%; height: 6px; background-color: rgba(255, 255, 255, 0.1); border-radius: 3px; overflow: hidden; margin-bottom: 10px; }
    .param-meter-fill { height: 100%; background: linear-gradient(90deg, var(--primary-gradient-start) 0%, var(--accent-color) 100%); border-radius: 3px; transition: width 0.5s ease-out; }
    /* 在侧边栏中使选择框标签为白色 */
    [data-testid="stSidebar"] .stSelectbox label { color: white !important; font-weight: 500; margin-bottom: 8px; }
    /* 隐藏原始聊天控件 */
    .stChatMessage { display: none !important; }

</style>
""",
    unsafe_allow_html=True,
)

# --- 初始化 Session State ---
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "model_loaded" not in st.session_state: 
    st.session_state.model_loaded = False
if "current_model_name" not in st.session_state: 
    st.session_state.current_model_name = None
if "gen_params" not in st.session_state:
    st.session_state.gen_params = {
        "temperature": 0.8, 
        "top_k": 50, 
        "top_p": 0.9, 
        "repetition_penalty": 1.0, 
        "frequency_penalty": 0.5
        }
if "load_error" not in st.session_state:
    st.session_state.load_error = None


# --- 侧边栏 ---
with st.sidebar:
    col1, col2 = st.columns([1, 5], vertical_alignment="bottom")
    with col1:
        st.image("./utils/Logo_gray.svg", width=60)
    with col2:
        st.title("Mini-LLM 控制面板")

    # 1. 模型选择与加载
    st.markdown("---")
    st.markdown('<p class="sidebar-section-header"><span>💾</span>模型选择与加载</p>', unsafe_allow_html=True)
    model_options_display = { name: f"{name} ({config.get('description', '无描述')})" for name, config in MODEL_CONFIG_DEFINITIONS.items() }
    # 获取模型名称
    def get_model_name_from_display(display_str): 
        parts = display_str.split(" (", 1)  # 参数 1 表示只分割一次
        return parts[0] if parts else display_str

    current_model_idx = 0
    if st.session_state.current_model_name:
        try: 
            current_model_idx = list(MODEL_CONFIG_DEFINITIONS.keys()).index(st.session_state.current_model_name)
        except ValueError: 
            current_model_idx = 0
    selected_display_option = st.selectbox(
        "🧠 选择一个模型:",
        options=list(model_options_display.values()), 
        index=current_model_idx, 
        key="model_select", 
        help="从列表中选择要加载和使用的语言模型。"
        )
    model_option = get_model_name_from_display(selected_display_option)
    load_button_placeholder = st.empty()
    load_status_placeholder = st.empty()

    if load_button_placeholder.button(f"加载 {model_option}", key="load_model_button", use_container_width=True, icon="🚀"):
        if st.session_state.model_loaded and st.session_state.current_model_name != model_option: 
            st.session_state.messages = []
        keys_to_clear = ["model", "config", "approx_params", "model_args", "device", "tokenizer", "current_model_name", "model_loaded", "load_error"]
        if st.session_state.current_model_name != model_option: 
            get_cached_model_structure.clear()
            load_cached_weights.clear()
            load_cached_tokenizer.clear()
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.model_loaded = False
        st.session_state.current_model_name = None
        loading_msg = f"正在准备加载 {model_option}..."
        load_status_placeholder.markdown(f"""<div style="padding: 10px; border-radius: 8px; background-color: rgba(255,255,255,0.1); margin: 10px 0;"><div style="display: flex; align-items: center; margin-bottom: 8px;"><div style="margin-right: 10px;">⏳</div><div style="font-weight: 500;">{loading_msg}</div></div><div class="progress-bar-container"><div class="progress-bar"></div></div></div>""", unsafe_allow_html=True)
        load_success = False
        load_start_time = time.time()
        try:
            selected_model_info = MODEL_CONFIG_DEFINITIONS.get(model_option)
            if not selected_model_info:
                error_msg = f"模型 '{model_option}' 的配置未找到！"
                st.toast(error_msg, icon="❌")
                st.session_state.load_error = f"配置缺失: {model_option}"
            else:
                model_path = selected_model_info["path"]
                yaml_path = selected_model_info["yaml"]
                inference_args = selected_model_info.get("inference_args", {})
                config_from_yaml = load_yaml(yaml_path)

                if config_from_yaml is None:
                    raise ValueError(f"无法从 '{yaml_path}' 加载 YAML 配置。")
                config_from_yaml["max_batch_size"] = 1
                config_from_yaml.update(inference_args)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, model_args = get_cached_model_structure(model_option, config_from_yaml)
                if model is None:
                    raise RuntimeError("加载模型结构失败。")
                loaded_model, load_weights_status = load_cached_weights(model, model_path, device)
                if load_weights_status != "success":
                    raise RuntimeError(f"加载权重失败: {load_weights_status}")
                tokenizer, load_tok_status = load_cached_tokenizer(TOKENIZER_NAME);
                if load_tok_status != "success":
                    raise RuntimeError(f"加载分词器失败: {load_tok_status}")
                load_end_time = time.time()
                approx_params = "未知"
                if hasattr(loaded_model, 'count_parameters'):
                    _, approx_params = loaded_model.count_parameters()
                st.session_state.update({ 
                    "model": loaded_model,
                    "config": config_from_yaml, 
                    "approx_params": approx_params, 
                    "model_args": model_args, 
                    "device": device, 
                    "tokenizer": tokenizer, 
                    "model_loaded": True, 
                    "current_model_name": model_option, 
                    "load_error": None })
                load_duration = load_end_time - load_start_time
                st.toast(f"模型 '{model_option}' 加载成功! ({load_duration:.2f}s)", icon="✅")
                load_success = True
        except FileNotFoundError as e:
            error_msg = f"文件加载失败: {e}"; st.toast(error_msg, icon="❌")
            st.session_state.load_error = error_msg
            st.session_state.model_loaded = False
        except (Exception, RuntimeError, ValueError) as e:
            error_msg = f"加载失败: {e}"
            st.toast(error_msg, icon="❌")
            st.session_state.load_error = error_msg
            st.session_state.model_loaded = False
            print(f"加载模型时出错: {traceback.format_exc()}")
        finally:
            load_status_placeholder.empty()
            if load_success:
                st.rerun()
            elif st.session_state.load_error:
                load_status_placeholder.markdown(f"""<div class="alert-box error pulse"><div style="margin-right: 10px;">💔</div><div style="font-weight: 500;">加载失败: {st.session_state.load_error}</div></div>""", unsafe_allow_html=True)
            if not load_success:
                 if load_button_placeholder.button(f"🔄 重试加载 {model_option}", key="reload_model_button_after_fail", use_container_width=True): st.rerun()
    
    # 2. 模型信息
    st.markdown("---")
    st.markdown('<p class="sidebar-section-header"><span>ℹ️</span>当前模型信息</p>', unsafe_allow_html=True)
    if st.session_state.get("model_loaded", False) and st.session_state.get("current_model_name"):
        display_name = st.session_state.current_model_name
        display_params = st.session_state.get("approx_params", "N/A")
        display_len = st.session_state.config.get("max_seq_len", "N/A") if "config" in st.session_state and st.session_state.config else "N/A"
        display_device = st.session_state.get("device", "N/A").upper()
        tok_vocab_size = st.session_state.tokenizer.vocab_size if st.session_state.get("tokenizer") and hasattr(st.session_state.tokenizer, 'vocab_size') else "N/A"
        st.markdown(f"""<div class="info-card"><h4><span>🧩</span>模型详情</h4><p><span>🏷️</span><strong>名称:</strong> {display_name}</p><p><span>⚙️</span><strong>参数 (约):</strong> {display_params}</p><p><span>↔️</span><strong>最大长度:</strong> {display_len} tokens</p><p><span>🗣️</span><strong>词汇量:</strong> {tok_vocab_size}</p><p><span>💻</span><strong>运行设备:</strong> {display_device}</p></div>""", unsafe_allow_html=True)
    elif st.session_state.get("load_error"):
        st.markdown(f"""<div class="alert-box error"><div style="margin-right: 10px;">⚠️</div><div style="font-weight: 500;">模型未加载。上次错误: {st.session_state.load_error}</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="alert-box info"><div style="margin-right: 10px;">👆</div><div style="font-weight: 500;">请在上方选择模型并点击 '加载模型'。</div></div>""", unsafe_allow_html=True)
    
    # 3. 生成参数
    st.markdown("---")
    st.markdown('<p class="sidebar-section-header"><span>🔧</span>生成参数调整</p>', unsafe_allow_html=True)
    model_loaded = st.session_state.get("model_loaded", False)
    with st.expander("展开调整生成参数", expanded=False):
        gen_params = st.session_state.gen_params
        temp = st.slider("🌡️ Temperature (随机性)", 0.01, 2.0, gen_params.get('temperature', 0.8), 0.01, help="控制生成文本的随机性。较低值更精确，较高值更具创意。", key="param_temp", disabled=not model_loaded)
        temp_text_color = "#a0aec0"; temp_label = "平衡";
        if temp < 0.5: 
            temp_label = "更精确"
            temp_text_color = "#63b3ed"
        elif temp > 1.2:
            temp_label = "更创意"
            temp_text_color = "#fc8181"
        st.markdown(f"""<div class="param-indicator"><span>0.01</span><span style="font-weight: 600; color: {temp_text_color};">{temp_label} ({temp:.2f})</span><span>2.00</span></div>""", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("🔝 Top-k (最佳token数)", 1, 200, gen_params.get('top_k', 50), 1, help="在每步生成时，仅考虑概率最高的 k 个token。", key="param_topk", disabled=not model_loaded)
        with col2:
            top_p = st.slider("🅿️ Top-p (核采样)", 0.1, 1.0, gen_params.get('top_p', 0.9), 0.01, help="选择累积概率超过 p 的最小token集进行采样 (0.1 到 1.0)。", key="param_topp", disabled=not model_loaded)
        col3, col4 = st.columns(2)
        with col3:
            rep_pen = st.slider("🔁 重复惩罚", 1.0, 2.0, gen_params.get('repetition_penalty', 1.0), 0.01, help="对重复出现的token施加惩罚 (大于 1.0 会减少重复，不可与频率惩罚同时使用)。", key="param_reppen", disabled=not model_loaded)
        with col4:
            freq_pen = st.slider("🔄 频率惩罚", 0.0, 2.0, gen_params.get('frequency_penalty', 0.5), 0.01, help="基于token在已生成文本中的频率降低其概率 (0 表示不惩罚，不可与重复惩罚同时使用)。", key="param_freqpen", disabled=not model_loaded)
        if model_loaded:
            creativity_score = max(0, min(1.0, (temp / 2.0 * 0.5) + (top_p * 0.3) + ((200 - top_k) / 200 * 0.2)))
            coherence_score = max(0, min(1.0, ((2.0 - temp) / 2.0 * 0.4 if temp > 0 else 0.4) + ((2.0 - (rep_pen - 1.0)) * 0.3) + ((2.0 - freq_pen)/ 2.0 * 0.3)))
            st.markdown("""<div style="background-color: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 10px 12px; margin-top: 15px; border: 1px solid rgba(255,255,255,0.1);"><div style="font-weight: 600; font-size: 0.9em; margin-bottom: 10px; color: white; text-align: center;">参数组合倾向 (估算)</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="margin-bottom: 8px;"><div class="param-indicator"><span>创造性</span><span>{creativity_score:.2f} / 1.0</span></div><div class="param-meter"><div class="param-meter-fill" style="width: {creativity_score*100}%; background: linear-gradient(90deg, #8b5cf6 0%, #ec4899 100%);"></div></div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div><div class="param-indicator"><span>精确性/连贯性</span><span>{coherence_score:.2f} / 1.0</span></div><div class="param-meter"><div class="param-meter-fill" style="width: {coherence_score*100}%; background: linear-gradient(90deg, #22d3ee 0%, #6ee7b7 100%);"></div></div></div></div>""", unsafe_allow_html=True)
        st.session_state.gen_params = { "temperature": temp, "top_k": top_k, "top_p": top_p, "repetition_penalty": rep_pen, "frequency_penalty": freq_pen }
    if not model_loaded:
        st.markdown("""<div class="alert-box warning" style="margin-top: 10px;"><div style="margin-right: 10px;">🔒</div><div style="font-weight: 500;">加载模型后可调整生成参数。</div></div>""", unsafe_allow_html=True)
    
    # 4. 对话控制
    st.markdown("---")
    st.markdown('<p class="sidebar-section-header"><span>⚙️</span>对话控制</p>', unsafe_allow_html=True)
    if st.button("清空对话历史", key="clear_chat", use_container_width=True, disabled=not st.session_state.messages, icon="🧹"):
        st.session_state.messages = []
        st.toast("对话历史已清空", icon="🧹")
        st.rerun()


# --- 对话界面 ---
col1, col2 = st.columns([1, 10], vertical_alignment="bottom")
with col1:
    st.image("./utils/logo_gray.svg", width=60)
with col2:
    st.title("Mini-LLM")

# --- 展示状态 ---
if st.session_state.get("model_loaded", False) and st.session_state.get("current_model_name"):
    temp_value = st.session_state.gen_params.get('temperature', 'N/A')
    temp_display = f"{temp_value:.2f}" if isinstance(temp_value, (float, int)) else temp_value
    st.markdown(f"""<div class="caption-text"><span style="font-weight: 600;">💡 当前模型:</span> {st.session_state.current_model_name} | <span style="font-weight: 600;">🌡️ 温度:</span> {temp_display} | <span style="font-weight: 600;">💻 运行设备:</span> {st.session_state.get("device", "N/A").upper()}</div>""", unsafe_allow_html=True)
elif st.session_state.get("load_error"):
    st.markdown(f"""<div class="caption-text" style="border-left-color: #ef4444; color: #fecaca;"><span style="font-weight: 600;">🚫 模型加载出错:</span> {st.session_state.load_error}</div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="caption-text" style="border-left-color: #f59e0b; color: #fed7aa;"><span style="font-weight: 600;">⚠️ 当前无模型加载，请在侧边栏加载模型。</span></div>""", unsafe_allow_html=True)


# --- 对话历史展示 ---
chat_display_container = st.container()
with chat_display_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            escaped_content = html.escape(content)
            st.markdown(f"""<div class="message-container user"><div class="avatar user">👤</div><div class="chat-bubble user-bubble"><span>{escaped_content}</span></div></div>""", unsafe_allow_html=True)
        elif role == "assistant":
            bubble_info_html = ""
            escaped_content = html.escape(content)
            st.markdown(f"""<div class="message-container assistant"><div class="avatar assistant">🤖</div><div class="chat-bubble assistant-bubble"><span>{escaped_content}</span>{bubble_info_html}</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- 处理对话输入输出 ---
prompt = st.chat_input( "💬 请在此输入您的问题或指令...", disabled=not st.session_state.get("model_loaded", False), key="chat_input" )

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    escaped_content = html.escape(prompt)
    st.markdown(f"""<div class="message-container user"><div class="avatar user">👤</div><div class="chat-bubble user-bubble"><span>{escaped_content}</span></div></div>""", unsafe_allow_html=True)
    streaming_placeholder = st.empty()

    try:
        # --- 获取必要的状态和配置 ---
        gen_params = st.session_state.gen_params
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer
        config = st.session_state.config
        device = st.session_state.device
        max_len = config.get("max_seq_len", 512) if config else 512

        # --- 准备 generate 函数的参数 ---
        generate_args = {
            "prompt": prompt,
            "context_length": max_len,
            "tokenizer": tokenizer,
            "stream": True,
            "task": "chat",
            "temperature": gen_params.get("temperature", 0.8),
            "top_k": gen_params.get("top_k", 50),
            "top_p": gen_params.get("top_p", 0.9),
            "repetition_penalty": gen_params.get("repetition_penalty", 1.0),
            "frequency_penalty": gen_params.get("frequency_penalty", 0.5),
        }
    
        try:
            sig = inspect.signature(model.generate)
            valid_args = {k: v for k, v in generate_args.items() if k in sig.parameters}
        except AttributeError:
             print("DEBUG: Could not inspect model.generate signature, using all args.")
             valid_args = generate_args

        response_gen_start_time = time.time()
        full_response = ""

        # --- 流式输出 ---
        output_generator = model.generate(**valid_args)

        # --- 流式输出优化 ---
        response_container = st.empty()
        full_response = ""
        
        # --- 循环更新 placeholder ---
        for chunk in output_generator:
            full_response += str(chunk)
            escaped_response = html.escape(full_response)
            
            # 使用增量更新方式，避免完全重绘
            response_container.markdown(f"""
            <div class="chat-container" style="gap: 0;">
                <div class="message-container assistant">
                    <div class="avatar assistant">🤖</div>
                    <div class="chat-bubble assistant-bubble">
                        <span>{escaped_response}</span>
                        <div class="cursor-blink"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 添加微小延迟，平衡流畅性和性能
            time.sleep(0.01)
            
        # 最终更新状态，避免频繁rerun
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        error_message = f"生成回复时出错: {str(e)}"; detailed_error = traceback.format_exc()
        st.error(error_message); print(f"Generation Error: {detailed_error}")
        try:
            streaming_placeholder.empty()
        except:
            pass
        st.rerun()


# --- 未加载模型时显示提示信息 ---
if not prompt and not st.session_state.get("model_loaded", False) and not st.session_state.get("load_error"):
    st.markdown("""<div class="info-card" style="text-align: center; padding: 2rem; margin-top: 2rem;"><h2>👈 请从侧边栏选择并加载模型</h2><p>选择一个可用的模型并点击 "加载模型" 按钮来开始对话吧！</p></div>""", unsafe_allow_html=True)