import json, os, threading, re, html
from datetime import datetime
from typing import Optional, List

import flet as ft
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# ---- Flet API shims ----
Icons = getattr(ft, "icons", None) or getattr(ft, "Icons")
Colors = getattr(ft, "colors", None) or getattr(ft, "Colors")

# ---- Prompt ----
PROMPT = """
You are a helpful tutor assistant for school students.
Your role is to help students with their studies, answer questions, explain concepts, 
and provide educational guidance across various subjects.

You should:
- Provide clear, age-appropriate explanations
- Help with homework and study questions
- Explain concepts step by step
- Be encouraging and supportive
- Adapt your teaching style to the student's level

Please structure your response in the following format:
ANSWER: [Your main response to the student's question]

EXPLANATION: [Additional detailed explanation if needed]

TIPS: [Helpful study tips or resources, one per line starting with •]

Keep your responses friendly, educational, and appropriate for the student's level.
""".strip()




# ---- TutorResponse schema ----
class TutorResponse(BaseModel):
    answer: str = Field(...)
    explanation: Optional[str] = Field(default=None)
    helpful_tips: Optional[List[str]] = Field(default=None)


def parse_response(text: str) -> TutorResponse:
    answer = text; explanation = None; helpful = None
    try:
        lines, cur = text.split("\n"), None
        a, ex, tips = [], [], []
        for raw in lines:
            line = raw.strip()
            if not line: continue
            up = line.upper()
            if up.startswith("ANSWER:"):
                cur = "a"; a.append(line[7:].strip())
            elif up.startswith("EXPLANATION:"):
                cur = "ex"; ex.append(line[12:].strip())
            elif up.startswith("TIPS:"):
                cur = "tips"; continue
            elif line[:1] in ("•","-","*"):
                if cur!="tips": cur="tips"
                tips.append("- " + line[1:].strip())  # ✅ keep as markdown bullet
            else:
                (a if cur in (None,'a') else ex if cur=='ex' else tips).append(line)
        if a: answer = "\n".join(a).strip()
        if ex: explanation = "\n".join(ex).strip()
        if tips: helpful = [t for t in tips if t]
    except Exception:
        pass
    return TutorResponse(answer=answer, explanation=explanation, helpful_tips=helpful)


# ---- Markdown renderer ----
def render_markdown(md_text: str) -> ft.Markdown:
    return ft.Markdown(
        md_text or "",
        selectable=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        code_theme="atom-one-dark",
    )


# ---- App ----
def main(page: ft.Page):
    # Palette (light only)
    BG = "#F4F6FB"; CARD="#FFFFFF"; BORDER="#E8ECF3"; TEXT_MUTED="#6B7280"
    GRAD_1="#7C3AED"; GRAD_2="#EC4899"; USER_TINT="#F3E8FF"
    SUCCESS=getattr(Colors,"GREEN_400","#66BB6A"); ERROR=getattr(Colors,"RED_400","#EF5350")

    page.title="AI Tutor — Local"
    page.theme_mode=ft.ThemeMode.LIGHT
    page.theme=ft.Theme(color_scheme_seed=GRAD_2, use_material3=True)
    page.bgcolor=BG; page.window_min_width=1100; page.window_min_height=720; page.padding=24

    # ---- Settings ----
    base_url = ft.TextField(label="Local API Base URL", value="http://localhost:11434/v1", expand=True, border_radius=12)
    model_name = ft.TextField(label="Model name", value="school-tutor:latest", expand=True, border_radius=12)

    def get_agent() -> Agent:
        model = OpenAIChatModel(model_name.value, provider=OpenAIProvider(base_url=base_url.value))
        return Agent(model=model, system_prompt=PROMPT)

    # ---- History ----
    HISTORY_FILE="tutor_history.json"; history=[]; current_chat_id=None; current_messages=[]

    def load_history():
        nonlocal history
        if os.path.exists(HISTORY_FILE):
            try: history=json.load(open(HISTORY_FILE,"r",encoding="utf-8"))
            except Exception: history=[]
        else: history=[]

    def save_history():
        try: json.dump(history, open(HISTORY_FILE,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception: pass

    def new_chat():
        nonlocal current_chat_id, current_messages
        current_messages=[]
        new_id=(max([c["id"] for c in history], default=0)+1) if history else 1
        history.insert(0, {"id":new_id,"title":"New chat","messages":[]})
        current_chat_id=new_id; refresh_history_list()
        chat_list.controls.clear(); page.update()

    def load_chat(cid:int):
        nonlocal current_chat_id, current_messages
        found=next((c for c in history if c["id"]==cid), None)
        if not found: return
        current_chat_id=cid; current_messages=found["messages"][:]
        chat_list.controls.clear()
        for m in current_messages:
            chat_list.controls.append(bubble(m["text"], m["role"]=="user"))
        page.update()

    def update_title_if_needed(first_text:str):
        nonlocal history
        t=first_text.strip().replace("\n"," ")
        if len(t)>48: t=t[:48]+"…"
        for c in history:
            if c["id"]==current_chat_id and (c["title"]=="New chat" or not c["title"].strip()):
                c["title"]=t; break

    def append_current(role:str, text:str):
        nonlocal current_messages
        current_messages.append({"role":role,"text":text})
        for c in history:
            if c["id"]==current_chat_id:
                c["messages"]=current_messages[:]; break
        save_history()

    # ---- Left rail ----
    def rail_btn(icon, tip, cb): return ft.IconButton(icon, tooltip=tip, on_click=cb)
    def nav_rail():
        return ft.Container(
            width=56, bgcolor=CARD, border_radius=16, border=ft.border.all(1,BORDER), padding=8,
            content=ft.Column(
                [
                    ft.Icon(Icons.SCHOOL, color=GRAD_1, size=22),
                    ft.Divider(color=BORDER),
                    rail_btn(Icons.ADD,"New chat",lambda e:new_chat()),
                    rail_btn(Icons.HISTORY,"History",lambda e:open_history()),
                    rail_btn(Icons.SETTINGS,"Settings",lambda e:open_settings()),
                ],
                alignment=ft.MainAxisAlignment.START, spacing=4
            )
        )

    # ---- Hero header ----
    def prompt_card(title:str):
        return ft.Container(
            bgcolor=CARD, border_radius=16, border=ft.border.all(1,BORDER), padding=16, width=260,
            content=ft.Row([ft.Icon(Icons.BOLT, color=GRAD_2), ft.Text(title, weight=ft.FontWeight.W_600)],
                           spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            on_click=lambda e:(setattr(message,"value",title), page.update())
        )

    def hero():
        grad=ft.LinearGradient(begin=ft.alignment.top_left,end=ft.alignment.bottom_right,colors=[GRAD_1,GRAD_2])
        return ft.Container(
            bgcolor=CARD, border_radius=24, border=ft.border.all(1,BORDER), padding=24,
            content=ft.Column([
                ft.Row([ft.ShaderMask(blend_mode=ft.BlendMode.SRC_IN, shader=grad,
                                      content=ft.Text("Hey there 👋", size=20, weight=ft.FontWeight.W_700))]),
                ft.Text("What can I help with?", size=40, weight=ft.FontWeight.W_900, color=GRAD_1),
                ft.Text("Hand-crafted prompts for you. Premium, top-shelf study help — no judgment 😊", color=TEXT_MUTED),
                ft.Row([prompt_card("Create a step-by-step study plan"),
                        prompt_card("Write a polite email to a teacher"),
                        prompt_card("Summarize this article in key points"),
                        prompt_card("Explain photosynthesis like I'm 10")],
                       spacing=12, wrap=True),
            ], spacing=8)
        )

    # ---- Chat bubbles ----
    chat_list = ft.ListView(expand=True, spacing=12, auto_scroll=True)

    def bubble(text: str, is_user: bool):
        bg = USER_TINT if is_user else CARD
        border_col = "#E9D5FF" if is_user else BORDER
        return ft.Row(
            [ft.Container(bgcolor=bg, border_radius=16, border=ft.border.all(1,border_col), padding=14, width=820,
                          content=ft.Column([
                              ft.Row([ft.Text("You" if is_user else "Tutor", weight=ft.FontWeight.W_700),
                                      ft.Text(datetime.now().strftime("%H:%M"), color=TEXT_MUTED, size=12)],
                                     alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                              render_markdown(text)], spacing=6, tight=True))],
            alignment=ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START
        )

    # ---- Composer ----
    message = ft.TextField(label="Message Tutor", multiline=True, min_lines=3, max_lines=6,
                           expand=True, border_radius=16)
    send_btn = ft.FilledButton("Ask Tutor", icon=Icons.SEND)

    def toast(msg, ok=True):
        page.snack_bar = ft.SnackBar(content=ft.Text(msg),
                                     bgcolor=SUCCESS if ok else ERROR,
                                     open=True, show_close_icon=True); page.update()

    # ---- Settings dialog ----
    def test_connection(_):
        import requests
        try:
            url=base_url.value.replace("/v1","/")
            r=requests.get(url,timeout=4)
            toast("Connection OK" if r.status_code==200 else "Connection failed", ok=(r.status_code==200))
        except Exception: toast("Connection failed", ok=False)

    settings_card = ft.Container(
        bgcolor=CARD, border_radius=16, border=ft.border.all(1,BORDER), padding=20, width=600,
        content=ft.Column([
            ft.Text("Local Model Configuration", weight=ft.FontWeight.W_600, size=16),
            ft.Divider(height=20, color=BORDER),
            base_url, model_name,
            ft.Row([
                ft.FilledButton("Test Connection", icon=Icons.MONITOR_HEART, on_click=test_connection),
                ft.OutlinedButton("Reset Defaults", icon=Icons.RESTART_ALT,
                                  on_click=lambda e:(setattr(base_url,"value","http://localhost:11434/v1"),
                                                     setattr(model_name,"value","school-tutor:latest"), page.update())),
            ], alignment=ft.MainAxisAlignment.END, spacing=12),
        ], spacing=16)
    )

    settings_dlg = ft.AlertDialog(
        modal=True, title=ft.Text("⚙️ Settings", weight=ft.FontWeight.BOLD, size=20),
        content=settings_card,
        actions=[ft.TextButton("Close", on_click=lambda e:(setattr(settings_dlg,"open",False), page.update()))],
        actions_alignment=ft.MainAxisAlignment.END
    )

    def open_settings(_=None):
        settings_dlg.open=True; page.update()

    page.overlay.append(settings_dlg)

    # ---- History dialog ----
    history_list = ft.Column(spacing=6, scroll=ft.ScrollMode.AUTO)

    def refresh_history_list():
        history_list.controls.clear()
        if not history:
            history_list.controls.append(ft.Text("No chats yet.", color=TEXT_MUTED))
        else:
            for chat in history:
                history_list.controls.append(
                    ft.ListTile(
                        title=ft.Text(chat["title"]), leading=ft.Icon(Icons.CHAT),
                        trailing=ft.IconButton(Icons.DELETE, tooltip="Delete",
                                               on_click=lambda e,cid=chat["id"]: delete_chat(cid)),
                        on_click=lambda e,cid=chat["id"]:(load_chat(cid), close_history())
                    )
                )
        page.update()

    def delete_chat(cid:int):
        nonlocal history, current_chat_id, current_messages
        history=[c for c in history if c["id"]!=cid]; save_history()
        if current_chat_id==cid:
            current_chat_id=None; current_messages=[]; chat_list.controls.clear()
        refresh_history_list()

    history_dlg = ft.AlertDialog(
        modal=True, title=ft.Text("🕘 Chat history", weight=ft.FontWeight.BOLD),
        content=ft.Container(width=480, bgcolor=CARD, border_radius=16,
                             border=ft.border.all(1,BORDER), padding=12, content=history_list),
        actions=[ft.TextButton("Close", on_click=lambda e: close_history())],
        actions_alignment=ft.MainAxisAlignment.END
    )

    def open_history():
        refresh_history_list(); history_dlg.open=True; page.update()
    def close_history():
        history_dlg.open=False; page.update()

    page.overlay.append(history_dlg)

    # ---- pubsub for thread-safe updates ----
    def on_pubsub(data):
        kind, payload = data
        if kind=="ok":
            chat_list.controls.append(bubble(payload, False))
            append_current("assistant", payload)
            message.value=""; send_btn.disabled=False; page.update()
        elif kind=="err":
            chat_list.controls.append(bubble(f"❌ Error: {payload}", False))
            append_current("assistant", f"❌ Error: {payload}")
            send_btn.disabled=False; page.update()
    page.pubsub.subscribe(on_pubsub)

    # ---- ask flow ----
    def do_ask():
        nonlocal current_chat_id
        q=(message.value or "").strip()
        if not q: toast("Please type a message", ok=False); return
        if current_chat_id is None: new_chat()
        if not current_messages: update_title_if_needed(q)

        chat_list.controls.append(bubble(q, True))
        append_current("user", q)
        page.update(); send_btn.disabled=True; page.update()

        def work():
            try:
                agent=get_agent()
                t0=datetime.now(); res=agent.run_sync(q); secs=(datetime.now()-t0).total_seconds()
                raw=str(getattr(res,"data",getattr(res,"output",res)))
                parsed=parse_response(raw)
                parts=[parsed.answer]
                if parsed.explanation: parts.append(f"\nExplanation:\n{parsed.explanation}")
                if parsed.helpful_tips: parts.append("\nTips:\n"+"\n".join([f"{t}" for t in parsed.helpful_tips]))
                parts.append(f"\n⏱ {secs:.2f}s")
                final="\n\n".join([p for p in parts if p and p.strip()])
                page.pubsub.send_all(("ok", final))
            except Exception as e:
                page.pubsub.send_all(("err", str(e)))
        threading.Thread(target=work, daemon=True).start()

    send_btn.on_click=lambda _: do_ask()

    # ---- layout ----
    page.appbar=ft.AppBar(
        title=ft.Row([ft.Icon(Icons.SCHOOL, color=GRAD_1), ft.Text("AI Tutor", weight=ft.FontWeight.BOLD)], spacing=8),
        bgcolor=BG, actions=[ft.IconButton(Icons.SETTINGS, tooltip="Settings", on_click=open_settings)],
        center_title=False, elevation=0,
    )
    hero_card=hero()
    composer=ft.Container(
        bgcolor=CARD, border_radius=16, border=ft.border.all(1,BORDER), padding=12,
        content=ft.Row([ft.Icon(Icons.ATTACH_FILE, color=GRAD_2), message, send_btn],
                       vertical_alignment=ft.CrossAxisAlignment.END, spacing=10)
    )

    load_history()
    page.add(
        ft.Row([
            nav_rail(), ft.Container(width=16),
            ft.Column([
                hero_card,
                ft.Container(height=12),
                ft.Container(bgcolor=CARD, border_radius=16, border=ft.border.all(1,BORDER),
                             padding=12, content=chat_list, expand=True),
                ft.Container(height=12),
                composer,
            ], expand=True, spacing=12)
        ], expand=True)
    )
    page.update()


ft.app(target=main)
