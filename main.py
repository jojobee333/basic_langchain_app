import os
from dotenv import load_dotenv
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

load_dotenv('lang.env')
# llm model specifications
open_ai_key = os.getenv('OPENAI_API_KEY')
model = "text-curie-001"

# temperature is the amount of randomness/creativity.
llm = OpenAI(temperature=0.9, model_name=model)
memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
messages = memory.chat_memory.messages

# title
st.title('GPT Creator')
default_prompt_text = 'Plug in your prompt here.'
# -----------------------------------------------
# prompt templates
prompt_templates = ('Generate 3 Social Media Content Ideas for ',
                    'Generate A Tiktok Social Media Caption for ')
# prompt selection widget
option = st.selectbox('Prompt', prompt_templates)
prompt_string = option + "{topic}"

# -----------------------------------------------
prompt = st.text_input(default_prompt_text)
st.write('Prompt Selected:', option)
if prompt:
    # set template variable to current prompt selection.
    main_template = PromptTemplate(
        input_variables=['topic'],
        template=prompt_string)
    # chain
    main_chain = LLMChain(llm=llm, prompt=main_template, memory=memory, verbose=True)
    response = main_chain.run(topic=prompt)
    st.write(response)
    with st.expander('Message History'):
        st.info(memory.buffer)

