# ingestion_script.py
import asyncio
import os
from pathlib import Path
import requests

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from raganything import RAGAnything, RAGAnythingConfig


def ollama_vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
    """Vision model function for multimodal processing - synchronous but works with LightRAG's wrapper"""
    vision_model = os.getenv("VISION_MODEL", "qwen2-vl:latest")
    ollama_host = os.getenv("LLM_BINDING_HOST", "http://host.docker.internal:11434")
    
    if messages:
        # Multimodal messages format (for RAG-Anything VLM enhanced queries)
        response = requests.post(
            f"{ollama_host}/api/chat",
            json={
                "model": vision_model,
                "messages": messages,
                "stream": False
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    
    elif image_data:
        # Single image format (for RAG-Anything image processing)
        chat_messages = []
        
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        
        # Add the image with the prompt
        chat_messages.append({
            "role": "user",
            "content": prompt,
            "images": [image_data] if isinstance(image_data, str) else image_data
        })
        
        response = requests.post(
            f"{ollama_host}/api/chat",
            json={
                "model": vision_model,
                "messages": chat_messages,
                "stream": False
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    
    else:
        # Text only - use regular LLM via LightRAG's built-in function
        # This will be handled by ollama_model_complete
        return ollama_model_complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            model_name=os.getenv("LLM_MODEL", "qwen2.5:latest"),
            host=ollama_host,
            options={"num_ctx": int(os.getenv("OLLAMA_LLM_NUM_CTX", "32768"))},
            **kwargs
        )


async def main():
    print("🚀 Starting RAG-Anything ingestion process...")
    
    # Get configuration from environment
    working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
    input_dir = Path(os.getenv("INPUT_DIR", "/app/data/inputs"))
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    ollama_host = os.getenv("LLM_BINDING_HOST", "http://host.docker.internal:11434")
    llm_model = os.getenv("LLM_MODEL", "qwen2.5:latest")
    embed_model = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
    
    print(f"📂 Working directory: {working_dir}")
    print(f"📥 Input directory: {input_dir}")
    print(f"🔢 Embedding dimension: {embedding_dim}")
    print(f"🤖 LLM Model: {llm_model}")
    print(f"📊 Embedding Model: {embed_model}")
    print(f"🌐 Ollama Host: {ollama_host}")
    
    # Initialize LightRAG instance with built-in Ollama functions
    print("⚙️  Initializing LightRAG with built-in Ollama support...")
    lightrag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,  # Use built-in function
        llm_model_name=llm_model,
        llm_model_kwargs={
            "host": ollama_host,
            "options": {"num_ctx": int(os.getenv("OLLAMA_LLM_NUM_CTX", "32768"))}
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: ollama_embed(  # Use built-in function
                texts,
                embed_model=embed_model,
                host=ollama_host
            )
        ),
    )
    
    print("🔄 Initializing storages...")
    await lightrag.initialize_storages()
    await initialize_pipeline_status()
    
    # Initialize RAG-Anything with vision model
    print("🎨 Initializing RAG-Anything with multimodal support...")
    rag = RAGAnything(
        lightrag=lightrag,
        vision_model_func=ollama_vision_func,  # Custom vision function
        config=RAGAnythingConfig(
            working_dir=working_dir,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
    )
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        return
    
    # Find all PDF files in input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {input_dir}")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF file(s) to process")
    
    # Process each PDF file
    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n{'='*60}")
        print(f"📄 Processing file {idx}/{len(pdf_files)}: {pdf_file.name}")
        print(f"{'='*60}")
        
        try:
            await rag.process_document_complete(
                file_path=str(pdf_file),
                output_dir="/app/data/outputs",
                parse_method="auto",
                display_stats=True
            )
            print(f"✅ Successfully processed: {pdf_file.name}")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("🎉 Ingestion process completed!")
    print("="*60)
    
    # Finalize storages
    await lightrag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())