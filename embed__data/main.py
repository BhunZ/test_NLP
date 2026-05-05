from hybrid_search import build_hybrid_retriever

if __name__ == "__main__":
    retriever = build_hybrid_retriever(
        data_path="/Users/carwyn/Downloads/transcript_v3.jsonl",
        k_retrieve=20,
        k_final=6,
    )

    # Test
    docs = retriever("What is gradient descent?")
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(doc.page_content[:300] + "...")
        print("Metadata:", doc.metadata)
    docs = retriever("What is gradient descent?")
    print("Retrieved docs:", len(docs))