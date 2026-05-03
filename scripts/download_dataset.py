import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data")


def download_squad(max_articles: int = 8, qa_per_article: int = 5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading SQuAD v2 from Hugging Face...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")

    articles = {}
    for row in ds:
        if not row["answers"]["text"]:
            continue

        title = row["title"]
        if title not in articles:
            articles[title] = {
                "title": title,
                "contexts": {},
                "qa_pairs": [],
            }

        context = row["context"]
        ctx_key = hash(context)
        if ctx_key not in articles[title]["contexts"]:
            articles[title]["contexts"][ctx_key] = context

        articles[title]["qa_pairs"].append({
            "question": row["question"],
            "answer": row["answers"]["text"][0],
        })

    selected = []
    for _, article in sorted(
        articles.items(),
        key=lambda x: len(x[1]["qa_pairs"]),
        reverse=True,
    ):
        if len(article["qa_pairs"]) >= qa_per_article:
            selected.append(article)
        if len(selected) >= max_articles:
            break

    manifest = []
    for article in selected:
        full_text = "\n\n".join(article["contexts"].values())
        filename = article["title"].replace(" ", "_").replace("/", "_")[:80] + ".txt"

        with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
            f.write(full_text)

        manifest.append({
            "title": article["title"],
            "filename": filename,
            "qa_pairs": article["qa_pairs"][:qa_per_article],
            "chars": len(full_text),
        })

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_q = sum(len(x["qa_pairs"]) for x in manifest)
    print(f"Prepared {len(manifest)} articles and {total_q} questions")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    download_squad()
