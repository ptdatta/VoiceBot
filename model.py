import chainlit as cl
import whisper
from langchain_community.llms import CTransformers
from transformers import pipeline
from io import BytesIO

whisper_model = whisper.load_model("small")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm


llm_model = load_llm()

def speech_to_text(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def summarize_text(text):
    if len(text.split()) < 100:
        return text
    summarized = summarizer(text, max_length=400, min_length=50, do_sample=False)
    return summarized[0]['summary_text']

def analyze_text_with_llama(text):
    summarized_text = summarize_text(text)
    prompt = f"""
Analyze the following text and describe the emotions or important points clearly.

Text:
{summarized_text}

Answer:
"""
    full_prompt = prompt
    response = llm_model.invoke(full_prompt)
    return response

audio_buffer = BytesIO()

@cl.on_chat_start
async def start():
    global audio_buffer
    audio_buffer = BytesIO()
    cl.user_session.set("audio_enabled", True)

    msg = cl.Message(content="🔵 Starting the Multi-modal Bot...")
    await msg.send()
    msg.content = "Hi! Send me **text**, **audio**, or use `/summarize` command!"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                audio_path = element.path
                await cl.Message(content="🔍 Converting audio file to text...").send()
                recognized_text = speech_to_text(audio_path)
                await cl.Message(content=f"📝 Transcribed Text: \n\n{recognized_text}").send()
                result = analyze_text_with_llama(recognized_text)
                await cl.Message(content=f"🤖 Analysis Result: \n\n{result}").send()
    else:
        user_text = message.content

        if user_text.startswith("/summarize"):
            text_to_summarize = user_text.replace("/summarize", "").strip()
            if not text_to_summarize:
                await cl.Message(content="⚠️ Please provide text after `/summarize` command.").send()
            else:
                summary = summarize_text(text_to_summarize)
                await cl.Message(content=f"📝 Summarized Text: \n\n{summary}").send()
        else:
            result = analyze_text_with_llama(user_text)
            await cl.Message(content=f"🤖 Analysis Result: \n\n{result}").send()

