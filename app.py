import os
import time
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=True)
os.environ["LANGSMITH_DISABLE_RUN_COMPRESSION"] = "true"

import gradio as gr
from app.rag.retriever import retriever_qa, ingest_pdf
from app.service.sql_service import ask_database
from app.database.supabase_client import save_chat
from app.utils.logger import logger

# In-memory session tracking for custom uploaded PDFs (max 2)
custom_pdfs = []

def respond(message, history, source, doc_option):
    if not message.strip():
        return "", history

    # Check gating for custom PDF mode
    if "PDF" in source and "Upload Your Own" in doc_option and len(custom_pdfs) == 0:
        bot_reply = (
            "🔒 **Upload Required:** You have selected 'Upload Your Own Custom Document', "
            "but haven't uploaded any PDF yet. Please upload a PDF in the panel above (max 2 allowed), "
            "or switch to **'Pre-Existing Enterprise Data'** to query the pre-loaded knowledge base immediately!"
        )
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_reply})
        return "", history

    start_time = time.time()
    try:
        if "PDF" in source:
            answer = retriever_qa(message)
            tool_used = "PDF Knowledge Base"
        else:
            answer = ask_database(message)
            tool_used = "SQL Database"
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Error: {err_msg}")
        if "429" in err_msg or "quota" in err_msg.lower() or "RESOURCE_EXHAUSTED" in err_msg:
            answer = "Your Gemini quota has finished, please check back later."
        else:
            answer = f"⚠️ Something went wrong: {err_msg}"

    end_time = time.time()
    resp_time = end_time - start_time

    # Log & persist to Supabase
    try:
        save_chat(question=message, answer=answer, tool_used=tool_used, response_time=resp_time)
    except Exception as log_err:
        logger.warning(f"Failed to log chat to Supabase: {log_err}")

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return "", history

def handle_pdf_upload(files):
    if not files:
        return "No files selected.", f"Active Custom Documents: {len(custom_pdfs)}/2", gr.update()

    if len(custom_pdfs) >= 2:
        return "⚠️ **Limit Reached:** Maximum 2 PDFs already active. Click 'Reset PDFs' to upload new ones.", f"Active Custom Documents: {len(custom_pdfs)}/2", gr.update()

    save_folder = PROJECT_ROOT / "Data" / "pdfs"
    save_folder.mkdir(parents=True, exist_ok=True)

    status_msgs = []
    file_list = files if isinstance(files, list) else [files]
    newly_indexed = []

    for file_obj in file_list:
        if len(custom_pdfs) >= 2:
            status_msgs.append("⚠️ Reached max 2 PDFs limit. Remaining files skipped.")
            break

        file_name = Path(file_obj.name).name
        if file_name in custom_pdfs:
            status_msgs.append(f"ℹ️ '{file_name}' already indexed.")
            continue

        dest_path = save_folder / file_name
        with open(file_obj.name, "rb") as src, open(dest_path, "wb") as dst:
            dst.write(src.read())

        try:
            chunks = ingest_pdf(str(dest_path))
            custom_pdfs.append(file_name)
            newly_indexed.append(file_name)
            status_msgs.append(f"✅ '{file_name}' indexed successfully ({chunks} chunks) into Qdrant Cloud!")
        except Exception as e:
            status_msgs.append(f"❌ Failed to index '{file_name}': {e}")

    active_summary = f"Active Custom Documents ({len(custom_pdfs)}/2): " + (", ".join([f"📄 {p}" for p in custom_pdfs]) if custom_pdfs else "None")

    if newly_indexed:
        chat_resume = [
            {
                "role": "assistant",
                "content": f"🎉 **Chat Resumed!** Successfully indexed **{', '.join(newly_indexed)}** into Qdrant Cloud. You can now ask any question about your document below!"
            }
        ]
    else:
        chat_resume = gr.update()

    return "\n".join(status_msgs), active_summary, chat_resume

def reset_custom_pdfs():
    global custom_pdfs
    custom_pdfs = []
    reset_chat = [
        {
            "role": "assistant",
            "content": "🔄 **Custom PDFs cleared.** Please upload a new PDF in the panel above to resume chat, or switch to Pre-Existing Enterprise Data."
        }
    ]
    return "🔄 Custom PDFs cleared.", "Active Custom Documents: 0/2", reset_chat

def on_doc_option_change(option):
    if "Upload Your Own" in option:
        privacy_text = "📤 **Custom Document Mode:** Please upload your PDF below (max 2 allowed) to index into Qdrant Cloud. Chat will unlock as soon as your PDF is ready!"
        if len(custom_pdfs) == 0:
            chat_state = [
                {
                    "role": "assistant",
                    "content": "🔒 **Upload your PDF to begin:** Upload your document in the panel above and click **'⚡ Index into Qdrant Cloud'**. Once indexed, chat will resume automatically!"
                }
            ]
        else:
            chat_state = [
                {
                    "role": "assistant",
                    "content": f"📄 **Active Custom Documents ({len(custom_pdfs)}/2):** {', '.join(custom_pdfs)}. Ask any question about your document below!"
                }
            ]
        return chat_state, privacy_text, gr.update(open=True)
    else:
        privacy_text = "🛡️ **Privacy Assured:** Currently querying pre-existing enterprise documents (Leave Policy & Employee Handbook). No personal files required."
        return [], privacy_text, gr.update(open=False)

custom_css = """
.gradio-container { font-family: 'Inter', system-ui, sans-serif; }
.header-box { text-align: center; padding: 20px; background: linear-gradient(135deg, #1e1b4b, #312e81); color: white; border-radius: 12px; margin-bottom: 20px; }
.privacy-badge { background: #064e3b; color: #a7f3d0; padding: 8px 14px; border-radius: 8px; font-size: 0.9em; font-weight: 500; margin-bottom: 12px; }
"""

with gr.Blocks(title="Enterprise Knowledge Agent", css=custom_css) as demo:
    with gr.Row():
        with gr.Column(scale=8):
            gr.Markdown(
                """
                # 🏢 Enterprise Knowledge Agent
                ### Dual-Engine Enterprise AI: Qdrant Cloud Vector Search (RAG) + Text-to-SQL
                🟢 **Vector Store:** Qdrant Cloud Active &nbsp;|&nbsp; 🟢 **Observability:** LangSmith Live Tracing &nbsp;|&nbsp; 🟢 **Audit DB:** Supabase
                """
            )
        with gr.Column(scale=2, min_width=160):
            gr.HTML(
                """
                <div style="display: flex; justify-content: flex-end; align-items: flex-start; padding-top: 14px;">
                    <a href="https://github.com/Gopal-next/Enterprise-Knowledege-Agent" target="_blank" style="display: inline-flex; align-items: center; gap: 8px; background: #24292e; color: #ffffff; padding: 9px 16px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
                        <svg height="20" width="20" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
                        GitHub Repo
                    </a>
                </div>
                """
            )

    with gr.Tabs():
        with gr.Tab("💬 Chat Assistant"):
            with gr.Row():
                source_radio = gr.Radio(
                    choices=["📄 PDF Knowledge Base", "🗄️ SQL Database"],
                    value="📄 PDF Knowledge Base",
                    label="Select Data Source",
                    interactive=True
                )
                doc_option_radio = gr.Radio(
                    choices=[
                        "🏢 Pre-Existing Enterprise Data (Demo / Privacy Mode)",
                        "📤 Upload Your Own Custom Document (Max 2 PDFs)"
                    ],
                    value="🏢 Pre-Existing Enterprise Data (Demo / Privacy Mode)",
                    label="Document Privacy / Source Option",
                    interactive=True
                )

            privacy_info = gr.Markdown(
                "🛡️ **Privacy Assured:** Currently querying pre-existing enterprise documents (Leave Policy & Employee Handbook). No personal files required."
            )

            # Upload section (expandable/visible for custom uploads)
            upload_accordion = gr.Accordion("📤 Upload Custom PDFs (Only needed for 'Upload Your Own' mode)", open=False)
            with upload_accordion:
                file_upload = gr.File(label="Upload PDF(s)", file_types=[".pdf"], file_count="multiple")
                with gr.Row():
                    upload_btn = gr.Button("⚡ Index into Qdrant Cloud", variant="primary")
                    reset_btn = gr.Button("🔄 Clear / Reset Custom PDFs", variant="secondary")
                upload_status = gr.Markdown("No custom files uploaded yet.")
                docs_count_status = gr.Markdown("Active Custom Documents: 0/2")

            chatbot = gr.Chatbot(height=450)

            upload_btn.click(handle_pdf_upload, inputs=[file_upload], outputs=[upload_status, docs_count_status, chatbot])
            reset_btn.click(reset_custom_pdfs, outputs=[upload_status, docs_count_status, chatbot])

            # Hook doc_option_radio to clear chat when custom mode is chosen
            doc_option_radio.change(
                on_doc_option_change,
                inputs=[doc_option_radio],
                outputs=[chatbot, privacy_info, upload_accordion]
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question (e.g. 'What is the leave policy?' or 'How many employees work in Engineering?')...",
                    show_label=False,
                    scale=9
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    ["What is the leave policy?", "📄 PDF Knowledge Base", "🏢 Pre-Existing Enterprise Data (Demo / Privacy Mode)"],
                    ["How many annual privilege leaves do employees get?", "📄 PDF Knowledge Base", "🏢 Pre-Existing Enterprise Data (Demo / Privacy Mode)"],
                    ["How many employees work in Engineering?", "🗄️ SQL Database", "🏢 Pre-Existing Enterprise Data (Demo / Privacy Mode)"],
                    ["List all employee departments and counts", "🗄️ SQL Database", "🏢 Pre-Existing Enterprise Data (Demo / Privacy Mode)"]
                ],
                inputs=[msg_input, source_radio, doc_option_radio]
            )

            # Submit handlers
            submit_btn.click(
                respond,
                inputs=[msg_input, chatbot, source_radio, doc_option_radio],
                outputs=[msg_input, chatbot]
            )
            msg_input.submit(
                respond,
                inputs=[msg_input, chatbot, source_radio, doc_option_radio],
                outputs=[msg_input, chatbot]
            )

        with gr.Tab("🏗️ Architecture & Stack"):
            gr.Markdown(
                """
                ### 🧱 System Architecture

                | Component | Technology | Role |
                |---|---|---|
                | **Vector Database** | **Qdrant Cloud** | Distributed high-dimensional vector search (3072-dim Cosine similarity) |
                | **Embeddings** | **Google Gemini (gemini-embedding-2)** | Dense document semantic embeddings |
                | **LLM Synthesis** | **Gemini 3.5 Flash** | Context-grounded RAG reasoning & Text-to-SQL synthesis |
                | **SQL Database** | **SQLite (SQLAlchemy)** | Relational data query engine for employee records |
                | **Observability** | **LangSmith** | Live distributed tracing (latency, tokens, retriever chunks) |
                | **Chat History** | **Supabase Cloud** | Cloud persistence for analytics and auditing |
                """
            )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)


