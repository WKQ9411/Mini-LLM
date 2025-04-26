import streamlit as st
from model import get_model_and_args # 假设 model.py 和这个函数存在且可用
import torch
from transformers import AutoTokenizer
import os # Import os for path joining if needed, although not used for the fixed path

# --- Configuration ---
# Define constants in the global scope
TOKENIZER_NAME = "./mini_tokenizer"
MODEL_NAME = "mini_llama3"  # <<< DEFINE MODEL_NAME HERE
MODEL_PATH = "C:\\Users\\WKQ\\Downloads\\pretrained_mini_llama3_epoch_1_iter_70000_loss_3.209038257598877-base.pt" # 保持用户指定路径

# 设置页面配置
st.set_page_config(
    page_title="Mini DeepSeek 聊天助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 模型初始化函数 (只运行一次)
@st.cache_resource
def load_model():
    """Loads the model and tokenizer."""
    # Use the globally defined constants
    tokenizer_name = TOKENIZER_NAME
    model_name = MODEL_NAME # Use global MODEL_NAME
    model_path = MODEL_PATH   # Use global MODEL_PATH

    # 加载tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        st.error(f"加载 Tokenizer 出错 ({tokenizer_name}): {e}")
        st.stop()
        return None, None, None # Match expected return values on error

    # 初始化模型
    try:
        Model, Model_Args = get_model_and_args(model_name) # Use model_name here
        model_args = Model_Args(max_batch_size=1, max_seq_len=256)
        model = Model(model_args)
        model.eval()
    except NameError:
        st.error(f"初始化模型出错: 'get_model_and_args' 未在 'model.py' 中找到或导入失败。")
        st.stop()
        return None, None, None
    except Exception as e:
        st.error(f"初始化模型出错 (模型名: {model_name}): {e}") # model_name is accessible here
        st.error("请确认 'model.py' 文件存在且包含正确的 'get_model_and_args' 函数及模型类。")
        st.stop()
        return None, None, None

    # 加载模型权重
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # It's often better to move model to device *before* loading state dict
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        # model.to(device) # Can be here or before load_state_dict
    except FileNotFoundError:
        st.error(f"模型权重文件未找到: {model_path}")
        st.error("请确认文件路径正确且文件存在。")
        st.stop()
        return None, None, None
    except Exception as e:
        st.error(f"加载模型权重或移动到设备出错 ({model_path}): {e}")
        st.stop()
        return None, None, None

    return model, tokenizer, device # Return device

# 加载模型
try:
    with st.spinner('正在加载模型，请稍候...'):
        # The function now only returns 3 items
        model, tokenizer, device = load_model()
        if model is None or tokenizer is None:
             st.error("模型或Tokenizer加载失败，应用无法启动。")
             st.stop() # 如果加载失败则停止
    st.success(f"模型加载成功! (运行于 {device.upper()})") # 显示运行设备
except Exception as e:
     st.error(f"加载模型过程中发生未预料的错误: {e}")
     st.stop()


# 侧边栏设置
with st.sidebar:
    st.title("模型设置")

    # 参数调节
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.05,
                           help="控制生成文本的随机性。较低的值使输出更确定，较高的值更随机。(建议 0.7-1.0)")
    top_k = st.slider("Top-k", min_value=1, max_value=100, value=50, step=1,
                     help="限制生成时只考虑概率最高的k个词。(建议 40-60)")
    top_p = st.slider("Top-p (Nucleus Sampling)", min_value=0.0, max_value=1.0, value=0.9, step=0.01,
                     help="限制生成时只考虑累积概率达到p的最小词集。与Top-k二选一或结合使用。(建议 0.85-0.95)")
    max_length = st.slider("最大生成长度", min_value=50, max_value=512, value=200, step=10,
                          help="限制单次回复生成的最大Token数量")

    # 模型信息
    st.divider()
    st.subheader("模型信息")
    approx_params_display = "未知"
    try:
        _, approx_params = model.count_parameters()
        approx_params_display = f"{approx_params:,}" if isinstance(approx_params, int) else str(approx_params)
    except AttributeError:
        approx_params_display = "无法获取 (方法不存在)"
    except Exception as e:
        approx_params_display = f"获取错误: {e}"

    st.write(f"模型参数量: {approx_params_display}")
    max_context = 512 # Keep hardcoded or retrieve if possible
    st.write(f"模型最大上下文长度: {max_context}")
    st.write(f"运行设备: {device.upper()}")

    st.divider()
    if st.button("清除聊天记录"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Mini DeepSeek 聊天助手 v1.0")

# 主界面
st.title("🤖 Mini DeepSeek 聊天助手")
# Now MODEL_NAME is accessible here because it's defined globally
st.caption(f"一个基于 {MODEL_NAME} 模型的对话助手")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        model_input_prompt = prompt # Still using only current prompt

        try:
            with st.spinner("思考中..."):
                output_generator = model.generate(
                    prompt=model_input_prompt,
                    context_length=max_length,
                    tokenizer=tokenizer,
                    stream=True,
                    task="generate",
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

                for output_chunk in output_generator:
                    if isinstance(output_chunk, str):
                        full_response += output_chunk
                        message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        except AttributeError as e:
             st.error(f"模型生成出错: 'generate' 方法可能不存在或参数不匹配。错误: {e}")
             st.error("请检查 model.py 中的 'generate' 函数定义及其接受的参数（是否包含 temperature, top_k, top_p, max_length 等）。")
             full_response = "抱歉，模型接口调用出错。"
             message_placeholder.markdown(full_response)
        except Exception as e:
             st.error(f"生成回复时发生意外错误: {e}")
             full_response = "抱歉，我在尝试生成回复时遇到了问题。"
             message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})