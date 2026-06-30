import time
import base64
import streamlit as st
from agent.react_agent import ReactAgent
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

st.title("智扫通机器人智能客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()


def set_background_with_opacity(image_path, opacity=0.5):
    """
    设置带有透明度（半透明）的背景图片
    :param image_path: 本地图片路径
    :param opacity: 图片透明度，0为完全透明，1为完全不透明
    """
    with open(image_path, "rb") as f:
        img_data = f.read()
    encoded = base64.b64encode(img_data).decode()
    bg_url = f"data:image/jpg;base64,{encoded}"

    st.markdown(
        f"""
        <style>
        /* 主容器定位为相对位置，以便伪元素绝对定位 */
        .stApp {{
            position: relative;
            background: none;  /* 移除原有背景 */
        }}
        /* 伪元素作为背景层 */
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url({bg_url}) no-repeat center center fixed;
            background-size: cover;
            opacity: {opacity};      /* 关键：设置图片透明度 */
            z-index: -1;             /* 置于底层 */
            pointer-events: none;    /* 让背景层不干扰鼠标事件 */
        }}
        /* 确保主要内容在背景层之上且有足够对比度，可选 */
        .stApp > div {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background_with_opacity(os.path.join(current_dir, "iu.jpg"), opacity=0.5)  # 调整透明度值

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({
        "role": "user", "content": prompt
    })
    response_messages = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)


        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)
                    yield char


        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({
            "role": "assistant", "content": response_messages[-1]
        })
