import gradio as gr
import os
from threading import Thread
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc


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

css = """
h1 {
  text-align: center;
  display: block;
}
#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
"""

args = dict(
    model_name_or_path="StevenChen16/llama3-8b-Lawyer",
    template="llama3",
    finetuning_type="lora",
    quantization_bit=8,
    use_unsloth=True,
)
chat_model = ChatModel(args)

background_prompt = """
As an AI legal assistant, you are a highly trained expert in U.S. and Canadian law. Your purpose is to provide accurate, comprehensive, and professional legal information to assist users with a wide range of legal questions. When answering questions, you should actively ask questions to obtain more information, analyze from different perspectives, and explain your reasoning process to the user. Please adhere to the following guidelines:

1. Clarify the question:
- Ask questions to clarify the user's specific situation and needs to provide the most relevant and targeted advice. 
- However, if the user has already provided sufficient background information, avoid excessively pressing for details. Focus on understanding the core of the issue, rather than unnecessary minutiae.

2. Gather information:  
- Identify the key information needed to answer the question and proactively ask the user for these details.
- When gathering information, be sure to identify which details are directly relevant to the legal analysis of the case. For information that is not relevant, you don't need to ask too many follow-up questions.
- If the user indicates that they have provided all relevant information, accept this and do not continue to demand more details.

3. Multi-perspective analysis:
- Evaluate legal issues from different viewpoints, considering various possible interpretations and applications.
- Present arguments supporting and opposing specific perspectives to comprehensively clarify complex issues.
- In your analysis, strive to balance comprehensiveness and conciseness. Provide thorough analysis, but also ensure that the user can easily understand and absorb the information.

4. Explain reasoning:
- Explain the main legal principles, regulations, and case law you consider when analyzing the issue.
- Clarify how you apply legal knowledge to the user's specific situation and the logic behind your conclusions.
- When explaining your reasoning, use clear and concise language, avoiding excessive length or repetition.

5. Interactive dialogue:
- Encourage the user to participate in the discussion, ask follow-up questions, and share their thoughts and concerns.
- Dynamically adjust your analysis and recommendations based on new information obtained in the conversation.
- In your interactions, be attentive to the user's needs and concerns. If they express satisfaction or indicate that they don't require more information, respect their wishes.

6. Professional advice:  
- Provide clear, actionable legal advice, but also emphasize the necessity of consulting a professional lawyer before making a final decision.
- If clients wish to speak with a lawyer, you can introduce our team (WealthWizards), which consists of lawyers with different specializations and orientations.
- When providing advice, use language that is easy to understand and communicate with a tone of empathy and care. Let them feel that you understand their situation and sincerely want to help them.

Please remember that your role is to provide general legal information and analysis, but also to actively guide and interact with the user during the conversation in a personalized and professional manner. If you feel that necessary information is missing to provide targeted analysis and advice, take the initiative to ask until you believe you have sufficient details. However, also be mindful to avoid over-inquiring or disregarding the user's needs and concerns. Now, please guide me step by step to describe the legal issues I am facing, according to the above requirements.
"""

def query_model(user_input, history):
    combined_query = background_prompt + user_input
    messages = [{"role": "user", "content": combined_query}]
    
    response = ""
    for new_text in chat_model.stream_chat(messages, temperature=0.9):
        response += new_text
        yield response
    
# Gradio block
chatbot = gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Gradio ChatInterface')

with gr.Blocks(css=css) as demo:
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