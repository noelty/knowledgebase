import os
import torch
import logging
import preprocess
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Load Qdrant Configuration from Environment
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Initialize Qdrant Client
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Load Sentence Transformer for Query Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load GPT-2 from Hugging Face
GPT2_MODEL_NAME = "gpt2"  # You can also use "gpt2-medium", "gpt2-large", "gpt2-xl" for larger versions
tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_NAME)
gpt2_model = AutoModelForCausalLM.from_pretrained(
    GPT2_MODEL_NAME,
    torch_dtype=torch.float16,  # Lower memory usage
    device_map="auto"  # Auto-select GPU if available
)

# Function to Generate Answer Using GPT-2
def generate_answer(query, context):
    """Generates a response using GPT-2 based on the retrieved context."""
    if not context.strip():
        return "I couldn't find relevant information."

    prompt = f"""
    Context: {context}

    Question: {query}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(gpt2_model.device)
    outputs = gpt2_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to Query Documents from Qdrant
def query_documents(collection_name, user_query, top_k=5, score_threshold=0.5):
    """Queries Qdrant, retrieves matching documents, and generates an answer using GPT-2."""
    try:
        logging.info(f"🔍 Original Query: {user_query}")
        processed_query = preprocess.preprocess_text(user_query)
        logging.info(f" Preprocessed Query: {processed_query}")

        # Generate Query Embedding
        query_vector = embedding_model.encode(processed_query).tolist()

        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        if not search_results:
            logging.warning(" No results found. Try increasing top_k or checking indexing.")

        # Filter Results
        filtered_results = [
            {
                "id": res.id,
                "score": res.score,
                "text": res.payload.get("text", ""),
            }
            for res in search_results if res.score >= score_threshold and "text" in res.payload
        ]

        # Extract Context for Answer Generation
        context = " ".join(res["text"] for res in filtered_results) or "No relevant information found."
        answer = generate_answer(user_query, context)

        return {"answer": answer, "chunks": filtered_results}

    except Exception as e:
        logging.error(f"Error during query: {e}")
        return {"error": str(e)}

# Command-Line Execution
if _name_ == "_main_":
    import argparse

    parser = argparse.ArgumentParser(description="Query documents with GPT-2")
    parser.add_argument("--collection", type=str, default="documents", help="Qdrant collection name")
    parser.add_argument("--query", type=str, required=True, help="Your search query")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()

    logging.info(f"Querying for: '{args.query}'")
    result = query_documents(args.collection, args.query, args.top_k)

    if "error" in result:
        logging.error(f" Error: {result['error']}")
    else:
        logging.info("\n===  Generated Answer ===")
        print(result["answer"])

        logging.info("\n===  Relevant Chunks ===")
        for i, chunk in enumerate(result["chunks"]):
            print(f"\nChunk {i+1} (Score: {chunk['score']:.3f}):\n{chunk['text']}")
