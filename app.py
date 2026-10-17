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
        return "No files selected.", f"Active Custom Documents: {len(custom_pdfs)}/2"

    if len(custom_pdfs) >= 2:
        return "⚠️ **Limit Reached:** Maximum 2 PDFs already active. Click 'Reset PDFs' to upload new ones.", f"Active Custom Documents: {len(custom_pdfs)}/2"

    save_folder = PROJECT_ROOT / "Data" / "pdfs"
    save_folder.mkdir(parents=True, exist_ok=True)

    status_msgs = []
    # Normalize files to list
    file_list = files if isinstance(files, list) else [files]

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
            status_msgs.append(f"✅ '{file_name}' indexed successfully ({chunks} chunks) into Qdrant Cloud!")
        except Exception as e:
            status_msgs.append(f"❌ Failed to index '{file_name}': {e}")

    active_summary = f"Active Custom Documents ({len(custom_pdfs)}/2): " + (", ".join([f"📄 {p}" for p in custom_pdfs]) if custom_pdfs else "None")
    return "\n".join(status_msgs), active_summary

def reset_custom_pdfs():
    global custom_pdfs
    custom_pdfs = []
    return "🔄 Custom PDFs cleared.", "Active Custom Documents: 0/2"

custom_css = """
.gradio-container { font-family: 'Inter', system-ui, sans-serif; }
.header-box { text-align: center; padding: 20px; background: linear-gradient(135deg, #1e1b4b, #312e81); color: white; border-radius: 12px; margin-bottom: 20px; }
.privacy-badge { background: #064e3b; color: #a7f3d0; padding: 8px 14px; border-radius: 8px; font-size: 0.9em; font-weight: 500; margin-bottom: 12px; }
"""

with gr.Blocks(title="Enterprise Knowledge Agent", css=custom_css) as demo:
    with gr.Column():
        gr.Markdown(
            """
            # 🏢 Enterprise Knowledge Agent
            ### Dual-Engine Enterprise AI: Qdrant Cloud Vector Search (RAG) + Text-to-SQL
            🟢 **Vector Store:** Qdrant Cloud Active &nbsp;|&nbsp; 🟢 **Observability:** LangSmith Live Tracing &nbsp;|&nbsp; 🟢 **Audit DB:** Supabase
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
            with gr.Accordion("📤 Upload Custom PDFs (Only needed for 'Upload Your Own' mode)", open=False):
                file_upload = gr.File(label="Upload PDF(s)", file_types=[".pdf"], file_count="multiple")
                with gr.Row():
                    upload_btn = gr.Button("⚡ Index into Qdrant Cloud", variant="primary")
                    reset_btn = gr.Button("🔄 Clear / Reset Custom PDFs", variant="secondary")
                upload_status = gr.Markdown("No custom files uploaded yet.")
                docs_count_status = gr.Markdown("Active Custom Documents: 0/2")

                upload_btn.click(handle_pdf_upload, inputs=[file_upload], outputs=[upload_status, docs_count_status])
                reset_btn.click(reset_custom_pdfs, outputs=[upload_status, docs_count_status])

            chatbot = gr.Chatbot(height=450)

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


