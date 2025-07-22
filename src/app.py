import streamlit as st
import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from graph import agent_graph, llm_for_streaming

st.set_page_config(page_title="民法典智能问答助手 (流式版)", page_icon="⚡", layout="wide")
st.title("⚡ 民法典智能问答助手 (流式版)")
st.caption("由DeepSeek, LangGraph, Milvus驱动")

if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None


def create_new_session():
    session_id = str(uuid.uuid4())
    now = datetime.now()
    session_name = f"对话于 {now.strftime('%Y-%m-%d %H:%M:%S')}"
    st.session_state.sessions[session_id] = {"name": session_name, "chat_history": []}
    st.session_state.current_session_id = session_id
    return session_id


def get_current_session():
    session_id = st.session_state.current_session_id
    return st.session_state.sessions.get(session_id)


with st.sidebar:
    st.header("会话管理")
    if st.button("➕ 开始新对话", use_container_width=True):
        create_new_session()
        st.rerun()
    st.markdown("---")
    if st.session_state.sessions:
        session_options = {sid: s["name"] for sid, s in st.session_state.sessions.items()}
        selected_session_id = st.selectbox(
            "选择一个历史对话",
            options=list(session_options.keys()),
            format_func=lambda sid: session_options[sid],
            index=list(session_options.keys()).index(st.session_state.current_session_id)
        )
        if selected_session_id != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_session_id
            st.rerun()
    else:
        st.info("点击上方按钮开始你的第一个对话吧！")

if not st.session_state.current_session_id:
    create_new_session()

current_session = get_current_session()

if current_session:
    for msg in current_session["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("请输入您关于民法典的问题..."):
    current_session["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_for_graph = []
    for msg in current_session["chat_history"]:
        if msg["role"] == "user":
            messages_for_graph.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages_for_graph.append(AIMessage(content=msg["content"]))

    graph_input = {"messages": messages_for_graph}

    with st.chat_message("assistant"):
        final_prompt = ""
        # 1. Agent思考过程，获取最终Prompt
        with st.status("正在思考...", expanded=True) as status:
            final_state = None
            for step_output in agent_graph.stream(graph_input, stream_mode="values"):
                final_state = step_output
                # 更新状态UI...
                if "condense_question" in final_state and final_state.get("condense_question"):
                    condensed_q = final_state['condense_question']
                    status.update(label=f"正在理解您的问题... (分析为: *{condensed_q}*)")
                if "intent" in final_state and final_state.get("intent"):
                    status.update(label=f"意图分析完成：**{final_state['intent']}**")
                if "rag_context" in final_state and final_state.get("rag_context"):
                    status.update(label="正在检索《民法典》知识库...")
                if "web_context" in final_state and final_state.get("web_context"):
                    status.update(label="正在进行网络搜索以获取最新信息...")
                if "final_prompt" in final_state and final_state.get("final_prompt"):
                    status.update(label="已准备好，正在生成回答...")

            if final_state and "final_prompt" in final_state:
                final_prompt = final_state["final_prompt"]

        # 2. 流式生成和显示回答
        if final_prompt:
            # 使用 st.write_stream流式输出
            response_stream = llm_for_streaming.stream(final_prompt)
            full_response = st.write_stream(response_stream)

            # 将完整的回答存入历史记录
            current_session["chat_history"].append({"role": "assistant", "content": full_response})
        else:
            error_msg = "抱歉，处理过程中出现了一点问题，没能准备好回答。"
            st.markdown(error_msg)
            current_session["chat_history"].append({"role": "assistant", "content": error_msg})