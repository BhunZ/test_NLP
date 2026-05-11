"""
cleanup_for_embedding.py
Xu ly transcript_hybrid_chunked.jsonl thanh embedding-ready JSONL

Xu ly:
1. Context Injection - them header (Course, Video, Source, Time)
2. Filler word removal - loai bo like, basically, you know, etc.
3. Add proper capitalization and punctuation

Chay: python cleanup_for_embedding.py
"""

import json
import re
from pathlib import Path

FILLER_WORDS = [
    r"\blike\b",
    r"\bbasically\b",
    r"\byou know\b",
    r"\bkind of\b",
    r"\bI think\b",
    r"\bI mean\b",
    r"\bso basically\b",
    r"\bwell\b",
    r"\byeah\b",
    r"\bokay\b",
    r"\bright\b",
    r"\bsure\b",
    r"\bactually\b",
    r"\breally\b",
    r"\bjust\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bprobably\b",
    r"\bcertainly\b",
    r"\bof course\b",
]

INPUT_FILE = Path(__file__).parent / "transcripts_hybrid_chunked.jsonl"
OUTPUT_FILE = Path(__file__).parent / "transcripts_enhanced.jsonl"


def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)

    for filler in FILLER_WORDS:
        text = re.sub(filler, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)

    replacements = [
        (r"\bWe spent sometimes\b", "The bots were trained"),
        (r"\btwo months training\b", "for months"),
        (r"\bOn thousands of CPUs\b", "using thousands of CPUs"),
        (r"\bterabytes of memory sometimes\b", "and terabytes of memory"),
        (r"\bwhen it came time to actually play\b", "when playing against humans"),
        (r"\bIt was just a lookup table\b", "It was just a lookup table"),
        (r"\bwhen they were in a tough spot\b", "when they were in a tough spot"),
        (r"\bThey would not act instantly, they would think\b", "they would think"),
        (r"\bThey would sit there and they would think\b", "they would sit and think"),
        (
            r"\bfor five seconds maybe five minutes\b",
            "for five seconds or five minutes",
        ),
        (r"\bit was a really difficult decision\b", "it was a difficult decision"),
        (
            r"\bit was clear that was allowing them to come up\b",
            "this allowed them to come up",
        ),
        (
            r"\bAnd so I wanted to investigate this behavior in our bots\b",
            "I wanted to investigate this behavior in our bots",
        ),
        (
            r"\bIf we could add this to our bots, how much of a difference would it make\b",
            "How much of a difference would adding this make",
        ),
        (
            r"\bThe ability to, instead of acting instantly\b",
            "The ability to think instead of acting instantly",
        ),
        (
            r"\bTo take some time and compute a better strategy\b",
            "to take some time and compute a better strategy",
        ),
        (
            r"\bFor the spot that the agent was in\b",
            "for the position the agent was in",
        ),
        (r"\bAnd this is what I found\b", "This is what I found"),
        (r"\bOn the x-axis here, we have\b", "On the x-axis"),
        (r"\byou can think of this as like the\b", "you can think of this as the"),
        (r"\bof parameters in your model\b", "of parameters in the model"),
        (r"\bOn the y-axis, we have\b", "On the y-axis"),
        (
            r"\bThis is basically like how much you would\b",
            "This is how much you would",
        ),
        (r"\bLose to a worst case adversary\b", "lose to a worst-case adversary"),
        (
            r"\bThe lower this number is, the better your poker bot is\b",
            "The lower this number is, the better the poker bot",
        ),
        (
            r"\band you can see, as you scale up the number of parameters\b",
            "As you scale up the number of parameters",
        ),
        (r"\bYour performance improves\b", "Performance improves"),
        (
            r"\bas you increase the number parameters by about 100x\b",
            "As you increase the parameters by about 100x",
        ),
        (
            r"\byour exploitability goes down by about half\b",
            "exploitability goes down by about half",
        ),
        (
            r"\bAnd you can see, as you increase the number parameters\b",
            "As you increase the parameters",
        ),
        (r"\byou can see, just adding search\b", "Just adding search"),
        (
            r"\bAdding the ability to sit there and think for a bit\b",
            "Adding the ability to think for a bit",
        ),
        (
            r"\bImproved the performance of these models\b",
            "improved the performance of these models",
        ),
        (r"\bIt reduced the exploitability\b", "It reduced exploitability"),
        (
            r"\bThe distance from Nash equilibrium by about 7x\b",
            "The distance from Nash equilibrium by about 7x",
        ),
        (
            r"\bif you were to extend that blue line and see how many\b",
            "To see how many",
        ),
        (
            r"\bParameters would you need in order to be comparable\b",
            "parameters would be needed to be comparable",
        ),
        (r"\bto adding search, the answer is\b", "to adding search, the answer is"),
        (r"\byou would need to scale up\b", "you would need to scale up"),
        (r"\bYour model by about 100,000x\b", "your model by about 100,000x"),
        (
            r"\bThis was pretty mind-blowing to me when I saw this\b",
            "This was mind-blowing when I saw this",
        ),
        (
            r"\bOver the course of my PhD, the first three years\b",
            "Over the first three years of my PhD",
        ),
        (
            r"\bfirst three or four years in my PhD, I managed to scale up\b",
            "I managed to scale up",
        ),
        (r"\bThese models by about 100x\b", "these models by about 100x"),
        (r"\bAnd I was proud of that\b", "I was proud of that"),
        (
            r"\bThat's a pretty impressive result, I think\b",
            "That is an impressive result",
        ),
        (
            r"\bBut what this plot was showing me was that just adding search\b",
            "But what the plot showed was that just adding search",
        ),
        (
            r"\bWas the equivalent of scaling things up by about 100,000x\b",
            "was equivalent to scaling up by about 100,000x",
        ),
        (
            r"\ball of my previous research up until this point\b",
            "All my previous research up until this point",
        ),
        (
            r"\bWould just be a footnote compared to adding search\b",
            "would just be a footnote compared to adding search",
        ),
        (r"\bWhen I saw this, it became clear\b", "When I saw this, it became clear"),
        (
            r"\bThis was the answer to beating top humans poker\b",
            "This was the answer to beating top humans in poker",
        ),
        (
            r"\bso for the next year, basically nonstop\b",
            "So for the next year, I worked nonstop",
        ),
        (r"\bI worked on scaling search\b", "I worked on scaling search"),
        (
            r"\bthere's a question that naturally comes up\b",
            "A question naturally comes up",
        ),
        (
            r"\bWhich is why wasn\'t this considered before\b",
            "Why was not this considered before",
        ),
        (r"\bFirst of all, I should say search had been\b", "First, search had been"),
        (r"\bConsidered in poker before\b", "considered in poker before"),
        (r"\bAnd it's actually quite natural to say, well\b", "It is natural to say"),
        (
            r"\bIf you had search in chess and search\b",
            "If you had search in chess and",
        ),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"\bthe bots the bots\b", "the bots", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s*\.\s*", ". ", text)
    text = re.sub(r"\s*,\s*,\s*", ",", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    result = []
    capitalize_next = True
    for word in words:
        if capitalize_next and word:
            word = word[0].upper() + word[1:] if len(word) > 1 else word.upper()
            capitalize_next = False
        if word and word[-1] in ".!?":
            capitalize_next = True
        result.append(word)

    text = " ".join(result)
    if text and text[-1] not in ".!?":
        text += "."

    return text


def create_context_header(record: dict) -> str:
    course = record.get("course", "Unknown")
    title = record.get("title", "Unknown")
    source = record.get("source", "Unknown")
    start = format_time(record.get("start_time", 0))
    end = format_time(record.get("end_time", 0))

    return (
        f"Course: {course}\nVideo: {title}\nSource: {source}\nTime: {start} - {end}\n\n"
    )


def process_chunk(record: dict) -> dict:
    cleaned_text = clean_text(record.get("chunk_text", ""))
    context_header = create_context_header(record)
    embedding_text = context_header + cleaned_text

    new_record = {
        "chunk_id": record.get("chunk_id", ""),
        "video_id": record.get("video_id", ""),
        "title": record.get("title", ""),
        "course": record.get("course", ""),
        "playlist_id": record.get("playlist_id", ""),
        "published_at": record.get("published_at", ""),
        "source": record.get("source", ""),
        "chunk_type": record.get("chunk_type", ""),
        "chunk_index": record.get("chunk_index", 0),
        "start_time": record.get("start_time", 0),
        "end_time": record.get("end_time", 0),
        "duration": record.get("duration", 0),
        "chunk_text": record.get("chunk_text", ""),
        "embedding_text": embedding_text,
        "word_count": record.get("word_count", 0),
        "token_estimate": record.get("token_estimate", 0),
    }

    return new_record


def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] File not found: {INPUT_FILE}")
        return

    output_records = []
    count = 0

    print(f"[INFO] Processing: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            new_record = process_chunk(record)
            output_records.append(new_record)
            count += 1
            if count % 5000 == 0:
                print(f"[INFO] Processed {count:,} chunks...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[SUCCESS] Output: {OUTPUT_FILE}")
    print(f"[INFO] Total chunks: {count:,}")


if __name__ == "__main__":
    main()
