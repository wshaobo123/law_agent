from typing import List, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import operator

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL_NAME
from tools import agent_tools


# 定义agent状态
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    condensed_question: str
    intent: str
    rag_context: str
    web_context: str
    final_prompt: str


# 大模型实例化
llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.1,
    streaming=True
)


# 用户问题query及历史缩写节点
def condense_question_node(state: AgentState):
    print("---NODE: 问题缩写---")
    if len(state["messages"]) == 1:
        condensed_question = state["messages"][-1].content
    else:
        history = "\n".join(
            [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in state["messages"][:-1]])
        latest_question = state["messages"][-1].content
        prompt = f"""请回顾以下对话历史和用户提出的最新问题。将它们合并成一个独立的、无需上下文就能理解的完整问题。
                    请直接返回这个完整问题，不要添加任何多余的解释或前缀。
                    对话历史:\n{history}\n\n最新问题: {latest_question}\n\n
                    凝练后的独立问题:"""
        response = llm.invoke(prompt)
        condensed_question = response.content.strip()
    print(f"缩写后的问题: {condensed_question}")
    return {"condensed_question": condensed_question}


# 意图分类节点，根据用户问题判断采用的工具
def classify_intent_node(state: AgentState):
    print("---NODE: 意图分类---")
    user_question = state["condensed_question"]
    system_prompt = f"""你是一个智能路由专家。根据用户的问题，判断最合适的处理方式。
                        你的选择有：'query_civil_code', 'web_search', 'both', 'general'。
                        用户问题: "{user_question}"
                        请只返回你的选择。"""
    response = llm.invoke(system_prompt)
    intent = response.content.strip()
    print(f"意图分类结果: {intent}")
    return {"intent": intent}


# RAG检索节点
def rag_retrieval_node(state: AgentState):
    print("---NODE: RAG检索---")
    user_question = state["condensed_question"]
    rag_tool = next(t for t in agent_tools if t.name == "query_civil_code")
    context = rag_tool.invoke(user_question)
    return {"rag_context": context}


# 网络搜索节点
def web_search_node(state: AgentState):
    print("---NODE: Web搜索---")
    user_question = state["condensed_question"]
    web_tool = next(t for t in agent_tools if t.name == "web_search")
    context = web_tool.invoke(user_question)
    return {"web_context": context}


# 准备提示词
def prepare_prompt_node(state: AgentState):
    print("---NODE: 准备最终提示词---")
    user_question = state["messages"][-1].content
    rag_context = state.get("rag_context", "")
    web_context = state.get("web_context", "")
    chat_history_str = "\n".join(
        [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in state["messages"][:-1]])

    final_prompt = f"""你是一个专业、严谨、友善的中国法律问答助手。
                    请根据以下提供的上下文信息和完整的对话历史，全面而清晰地回答用户【最新】的问题。
                    规则:
                    1. 优先使用并明确引用【法条原文】中的内容。
                    2. 如果【网络搜索结果】提供了补充信息（如最新案例、司法解释），请结合法条进行说明，但要指明信息来源。
                    3. 你的回答需要与【对话历史】保持连贯。
                    4. 如果上下文信息不足以回答问题，请坦诚告知，不要杜撰。
                    5. 答案应清晰、分点、易于理解，并首先给出核心结论。
                    6. 避免使用“根据我的知识”等模糊表述。
                    -------------
                    【对话历史】:
                    {chat_history_str if chat_history_str else "无"}
                    -------------
                    【法条原文】:
                    {rag_context if rag_context else "无"}
                    -------------
                    【网络搜索结果】:
                    {web_context if web_context else "无"}
                    -------------
                    用户【最新】的问题: "{user_question}"
                    """
    return {"final_prompt": final_prompt}


# 条件路由节点
def should_route(state: AgentState):
    """
    条件路由函数：根据意图分类的结果决定下一步走向。
    """
    print("---EDGE: 条件路由---")
    intent = state.get("intent")
    if intent == "query_civil_code":
        return "rag"  # 民法典问题直接走检索节点
    elif intent == "web_search":
        return "web"  # 需要网络搜索
    elif intent == "both":
        return "rag"  # both意图先走RAG
    else:  # 直接生成
        return "prepare_prompt"


def after_rag_route(state: AgentState):
    """
    处理both意图的特殊路由，在RAG检索后决定是否继续Web搜索。
    """
    print("---EDGE: RAG后路由---")
    # --- 核心修正点 2 ---
    if state.get("intent") == "both":
        return "web"  # 如果是both意图，RAG之后去Web
    else:
        return "prepare_prompt"  # 否则直接去准备提示词


# 图构建
def build_graph():
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("condense_question", condense_question_node)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("prepare_prompt", prepare_prompt_node)
    workflow.set_entry_point("condense_question")

    # 添加边
    workflow.add_edge("condense_question", "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_route,
        {"rag": "rag_retrieval", "web": "web_search", "prepare_prompt": "prepare_prompt"}
    )
    workflow.add_conditional_edges(
        "rag_retrieval",
        after_rag_route,
        {"web": "web_search", "prepare_prompt": "prepare_prompt"}
    )
    workflow.add_edge("web_search", "prepare_prompt")
    workflow.add_edge("prepare_prompt", END)

    return workflow.compile()


agent_graph = build_graph()
llm_for_streaming = llm