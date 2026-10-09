import gradio as gr

from core.ron import Ron

ron = Ron()


def respond(message, history):
    if not message or not message.strip():
        return ""
    return ron.chat(message.strip())


demo = gr.ChatInterface(
    fn=respond,
    api_name="chat",
    title="Ron-1",
    description="المساعد المستقل — يعمل بعقل Qwen3-1.7B محليًا.",
    textbox=gr.Textbox(placeholder="اكتب رسالتك إلى رون...", container=True),
    examples=["مرحبا يا رون", "من أنت؟", "ما عاصمة مصر؟"],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
