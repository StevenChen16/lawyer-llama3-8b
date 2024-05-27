import gradio as gr
import requests

# 描述
DESCRIPTION = '''
<div>
<h1 style="text-align: center;">AI Lawyer</h1>
</div>
'''

LICENSE = """
<p/>
---
Built with model "StevenChen16/Llama3-8B-Lawyer", based on "meta-llama/Meta-Llama-3-8B"
"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">AI Lawyer</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Ask me anything about US and Canada law...</p>
</div>
"""

# 定义前端Gradio调用后端API的函数
def query_model(user_input, history):
    url = "https://stevenchen16-llama3-8b-lawyer.hf.space/predict"  # 后端Flask API的URL
    payload = {
        "inputs": user_input
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers, stream=True)
    
    response_text = ""
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            response_text += chunk.decode('utf-8')
            yield response_text

# Gradio前端界面
chatbot = gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Gradio ChatInterface')

with gr.Blocks(theme='bethecloud/storj_theme') as demo:
    gr.Markdown(DESCRIPTION)
    gr.ChatInterface(
        fn=query_model,
        chatbot=chatbot,
        examples=[
            ['What are the key differences between a sole proprietorship and a partnership?'],
            ['What legal steps should I take if I want to start a business in the US?'],
            ['Can you explain the concept of "duty of care" in negligence law?'],
            ['What are the legal requirements for obtaining a patent in Canada?'],
            ['How can I protect my intellectual property when sharing my idea with potential investors?']
        ],
        cache_examples=False,
    )
    gr.Markdown(LICENSE)

if __name__ == "__main__":
    demo.launch(share=True)
