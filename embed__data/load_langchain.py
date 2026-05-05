
from langchain_core.documents import Document
import json

def load_documents(path: str):
    docs = []

    with open(path) as f:
        for line in f:
            chunk = json.loads(line)

            doc = Document(
                page_content=chunk["chunk_text"],
                metadata={
                    "video_id": chunk["video_id"],
                    "title": chunk["title"],
                    "course": chunk["course"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "duration": chunk["duration"],
                    "source": chunk["source"],
                    "chunk_type": chunk["chunk_type"],
                    "token_count": chunk["token_count"],
                }
            )

            docs.append(doc)

    return docs