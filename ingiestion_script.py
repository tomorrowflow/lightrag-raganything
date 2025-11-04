# ingestion_script.py
import asyncio
import os
from pathlib import Path
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from raganything import RAGAnything, RAGAnythingConfig
import requests
import base64

async def main():
    # Initialize LightRAG instance
    lightrag = LightRAG(
        working_dir=os.getenv("WORKING_DIR", "/app/data/rag_storage"),
        llm_model_func=ollama_llm_func,
        embedding_func=ollama_embedding_func,
    )
    
    await lightrag.initialize_storages()
    await initialize_pipeline_status()
    
    # Initialize RAG-Anything with vision model
    rag = RAGAnything(
        lightrag=lightrag,
        vision_model_func=ollama_vision_func,
        config=RAGAnythingConfig(
            working_dir=os.getenv("WORKING_DIR", "/app/data/rag_storage"),
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
    )
    
    # Process documents from input directory
    input_dir = Path(os.getenv("INPUT_DIR", "/app/data/inputs"))
    for pdf_file in input_dir.glob("*.pdf"):
        print(f"Processing {pdf_file.name} with RAG-Anything...")
        await rag.process_document_complete(
            file_path=str(pdf_file),
            output_dir="/app/data/outputs",
            parse_method="auto"
        )

def ollama_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    response = requests.post(
        f"{os.getenv('LLM_BINDING_HOST')}/api/generate",
        json={
            "model": os.getenv("LLM_MODEL"),
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def ollama_vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
    if messages:
        # Multimodal messages format
        response = requests.post(
            f"{os.getenv('LLM_BINDING_HOST')}/api/chat",
            json={
                "model": os.getenv("VISION_MODEL"),
                "messages": messages,
                "stream": False
            }
        )
        return response.json()["message"]["content"]
    elif image_data:
        # Single image format
        response = requests.post(
            f"{os.getenv('LLM_BINDING_HOST')}/api/generate",
            json={
                "model": os.getenv("VISION_MODEL"),
                "prompt": prompt,
                "images": [image_data],
                "stream": False
            }
        )
        return response.json()["response"]
    else:
        # Text only - use regular LLM
        return ollama_llm_func(prompt, system_prompt, history_messages, **kwargs)

def ollama_embedding_func(texts):
    embeddings = []
    for text in texts:
        response = requests.post(
            f"{os.getenv('EMBEDDING_BINDING_HOST')}/api/embeddings",
            json={
                "model": os.getenv("EMBEDDING_MODEL"),
                "prompt": text
            }
        )
        embeddings.append(response.json()["embedding"])
    return embeddings

if __name__ == "__main__":
    asyncio.run(main())