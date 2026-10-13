import os
from datetime import datetime, timezone
from uuid import uuid4
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

try:
    from database import messages_collection, memories_collection
except ImportError:
    st.error("Could not import collections from database.py.")
    st.stop()

st.set_page_config(page_title="ChatGPT with Memory & Deletion", page_icon="💬", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .stChatMessage { max-width: 800px; margin: 0 auto; }
        .stChatInput { max-width: 800px; margin: 0 auto; }
        div[data-testid="stVerticalBlock"] > div:has(div.stButton) {
            gap: 0px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is missing. Add it to your .env file.")
    st.stop()

chat_model = ChatGroq(model="openai/gpt-oss-20b", api_key=api_key, temperature=0.3)
uid = "user_101"


class MemItem(BaseModel):
    category: Literal["fact", "preference", "experience", "goal", "feeling", "belief", "other", "project"] = Field(description="The kind of user-specific information.")
    key: str = Field(description="A stable, short snake_case identifier for one topic. Reuse an existing key when updating.")
    value: str = Field(description="A concise, accurate statement of the user's information.")
    importance: float = Field(ge=0.0, le=1.0, description="How useful this info is.")

class MemDecision(BaseModel):
    should_store: bool = Field(description="True ONLY if the latest message contains useful USER-SPECIFIC long-term information.")
    memories: list[MemItem] = Field(description="Only memories worth saving or updating; empty if none.")

try:
    mem_extractor = chat_model.with_structured_output(MemDecision)
except Exception as e:
    st.error(f"Failed to init structured output: {e}")
    st.stop()


def fetch_mems(user_id: str) -> list[dict]:
    return list(memories_collection.find({"user_id": user_id}))

def fetch_history(conv_id):
    if not conv_id:
        return [], []

    db_msgs = list(messages_collection.find({"user_id": uid, "conversation_id": conv_id}).sort("timestamp", 1))
    
    lc_hist = []
    ui_hist = []

    for m in db_msgs:
        r = m.get("role", "user")
        c = m.get("content", "")
        ui_hist.append({"role": r, "content": c})

        if r == "user":
            lc_hist.append(HumanMessage(content=c))
        elif r == "assistant":
            lc_hist.append(AIMessage(content=c))

    return lc_hist, ui_hist

def store_msg(sess_id, conv_id, role, text):
    messages_collection.insert_one({
        "user_id": uid,
        "session_id": sess_id,
        "conversation_id": conv_id,
        "role": role,
        "content": text,
        "timestamp": datetime.now(timezone.utc),
    })

def get_chat_list(user_id):
    pipe = [
        {"$match": {"user_id": user_id}},
        {"$sort": {"timestamp": 1}},
        {"$group": {"_id": "$conversation_id", "first_msg": {"$first": "$content"}, "last_ts": {"$last": "$timestamp"}}},
        {"$sort": {"last_ts": -1}},
    ]
    return list(messages_collection.aggregate(pipe))

def remove_chat(conv_id):
    messages_collection.delete_many({"user_id": uid, "conversation_id": conv_id})


MEM_PROMPT = """
You are a strict long-term memory manager for a personal assistant.
Your ONLY job is to extract durable, personal facts about the user.

CRITICAL RULE: NEVER STORE GENERAL INFORMATION OR WORLD KNOWLEDGE.
- If the user asks "What is the capital of France?", return should_store=false.
- If the user says "Hello, how are you?", return should_store=false.
- If the user asks for a recipe, code fix, or explanation, return should_store=false.
- If the user is just chatting or exploring a topic, return should_store=false.

STORE ONLY information explicitly tied to the user's identity, such as:
- Personal facts (Name, location, job, age).
- Durable preferences ("I prefer Python over Java", "I am vegan").
- Ongoing projects or long-term goals.

UPDATES AND DUPLICATES:
- If the user corrects a stored fact (e.g., changes their name), return the EXACT same key with the new value.
- Keep keys short and snake_case (e.g., user_name, user_location).
"""

def format_mem_str(mems: list[dict]) -> str:
    lines = []
    for m in mems:
        k = m.get("key")
        v = m.get("value") or m.get("memory", "")
        if not v:
            continue
        if k:
            lines.append(f"- key={k!r}; value={v!r}")
        else:
            lines.append(f"- LEGACY: {v!r}")
    return "\n".join(lines) if lines else "No usable existing memories."

def get_new_mems(user_msg, past_chat, curr_mems):
    hist_str = "\n".join(
        f"{m.type}: {m.content}" for m in past_chat[-8:] if isinstance(m.content, str)
    ) or "No recent conversation context."

    mem_str = format_mem_str(curr_mems)

    prompt = f"""
EXISTING USER MEMORY PROFILE:
{mem_str}

RECENT CONVERSATION BEFORE THE LATEST MESSAGE:
{hist_str}

LATEST USER MESSAGE:
{user_msg}

Extract ONLY personal, durable information about the user. If the message is a general question, a greeting, or world knowledge, return should_store=false.
"""
    try:
        res = mem_extractor.invoke([
            ("system", MEM_PROMPT),
            ("human", prompt),
        ])

        if not isinstance(res, MemDecision) or not res.should_store or not res.memories:
            return MemDecision(should_store=False, memories=[])

        valid_items = []
        bad_words = ("api_key=", "api key:", "password=", "password:", "secret=", "token=")
        
        for i in res.memories:
            k = i.key.strip().lower().replace(" ", "_")
            v = i.value.strip()
            if not k or not v or any(w in v.lower() for w in bad_words):
                continue
            valid_items.append(MemItem(category=i.category, key=k, value=v, importance=i.importance))

        return MemDecision(should_store=bool(valid_items), memories=valid_items)

    except Exception as e:
        print(f"Memory extraction failed: {e}")
        return None

def save_mem_decision(user_id: str, dec):
    if not dec or not dec.should_store or not dec.memories:
        return 0

    cnt = 0
    now_time = datetime.now(timezone.utc)

    try:
        for item in dec.memories:
            k = item.key.strip().lower().replace(" ", "_")
            v = item.value.strip()

            memories_collection.update_one(
                {"user_id": user_id, "key": k},
                {
                    "$set": {"category": item.category, "value": v, "importance": item.importance, "updated_at": now_time},
                    "$setOnInsert": {"user_id": user_id, "key": k, "created_at": now_time},
                },
                upsert=True,
            )
            cnt += 1
        return cnt
    except PyMongoError as e:
        print(f"DB error saving memory: {e}")
        return 0

def build_mem_context(mems: list[dict]) -> str:
    struct_mems = [m for m in mems if m.get("key") and m.get("value")]
    if struct_mems:
        struct_mems.sort(key=lambda m: float(m.get("importance", 0.0) or 0.0), reverse=True)
        return "\n".join(f"- {m.get('key')}: {m.get('value')}" for m in struct_mems[:30])
    
    old_mems = [m.get("memory", "") for m in mems if m.get("memory")]
    return "\n".join(f"- {v}" for v in old_mems[:10]) or "No memories available."


if "sess_id" not in st.session_state:
    st.session_state["sess_id"] = str(uuid4())

if "curr_chat" not in st.session_state:
    st.session_state["curr_chat"] = None

with st.sidebar:
    st.title("💬 ChatGPT with Memory")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state["curr_chat"] = None
        st.rerun()

    st.divider()
    st.subheader("Recent Conversations")

    try:
        chat_list = get_chat_list(uid)
    except PyMongoError as e:
        st.error(f"Error loading chats: {e}")
        chat_list = []

    if not chat_list:
        st.caption("No conversations yet. Start typing to create one!")
    else:
        for c in chat_list:
            cid = c["_id"]
            title_txt = c.get("first_msg") or "Empty Conversation"
            short_title = title_txt[:30] + ("..." if len(title_txt) > 30 else "")
            is_active = st.session_state["curr_chat"] == cid

            if is_active:
                btn_col, del_col = st.columns([0.82, 0.18])
                with btn_col:
                    st.button(f"👉 💬 {short_title}", key=f"act_{cid}", use_container_width=True)
                with del_col:
                    if st.button("🗑️", key=f"rm_{cid}", help="Delete chat permanently"):
                        remove_chat(cid)
                        st.session_state["curr_chat"] = None
                        st.rerun()
            else:
                if st.button(f"💬 {short_title}", key=f"inact_{cid}", use_container_width=True):
                    st.session_state["curr_chat"] = cid
                    st.rerun()

    st.divider()
    with st.expander("🧠 Stored Long-Term Memories"):
        try:
            db_mems = fetch_mems(uid)
        except PyMongoError as e:
            st.error(f"Error loading memories: {e}")
            db_mems = []

        if db_mems:
            for m in db_mems:
                k = m.get("key", "legacy_memory")
                v = m.get("value", m.get("memory", ""))
                cat = m.get("category", "legacy")
                if v:
                    st.markdown(f"**{k}**: {v} *({cat})*")
        else:
            st.caption("No memories extracted yet.")


chat_id = st.session_state["curr_chat"] or str(uuid4())
lc_history, display_history = fetch_history(chat_id)

if not display_history:
    st.title("How can I help you today?")
else:
    for msg in display_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_text = st.chat_input("Message ChatGPT...")

if user_text:
    if st.session_state["curr_chat"] is None:
        st.session_state["curr_chat"] = chat_id

    chat_id = st.session_state["curr_chat"]

    try:
        store_msg(st.session_state["sess_id"], chat_id, "user", user_text)
        with st.chat_message("user"):
            st.markdown(user_text)

        lc_history, _ = fetch_history(chat_id)

        curr_mems = fetch_mems(uid)
        mem_ctxt = build_mem_context(curr_mems)

        sys_msg = SystemMessage(
            content=(
                "You are a helpful assistant. The user's CURRENT verified profile data is provided below. \n\n"
                "CRITICAL: If the older conversation history contradicts these stored memories, you MUST IGNORE the history and ONLY use these updated stored memories.\n\n"
                f"Stored user memories:\n{mem_ctxt}"
            )
        )

        old_msgs = lc_history[:-1]
        last_msg = lc_history[-1]
        
        reminder = SystemMessage(
            content=f"Reminder to assistant: Always use the latest stored memory facts. If the history says otherwise, the history is outdated. Current memory:\n{mem_ctxt}"
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = chat_model.invoke([sys_msg] + old_msgs + [reminder, last_msg])
                ai_reply = res.content
            st.markdown(ai_reply)

        store_msg(st.session_state["sess_id"], chat_id, "assistant", ai_reply)

        past_chat = lc_history[:-1]
        dec = get_new_mems(user_text, past_chat, curr_mems)
        
        changes = save_mem_decision(uid, dec)
        if changes:
            st.toast(f"Long-term memory saved/updated ({changes}).", icon="🧠")

    except PyMongoError as e:
        st.error(f"MongoDB operation failed: {e}")
    except Exception as e:
        st.error(f"Something went wrong while processing this message: {e}")

    st.rerun()