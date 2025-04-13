import os
import json

def generate_QA_pair_with_context(paper_number, publication_year, short_id, title, qa_pairs, context, save_dir):
    # Attach the same context to every QA pair
    enriched_qa_pairs = []
    for pair in qa_pairs:
        enriched_qa_pairs.append({
            "question": pair["question"],
            "answer": pair["answer"],
            "context": context.strip()
        })

    data = {
        "paper_id": f"{paper_number}_{publication_year}_{short_id}",
        "title": title,
        "qa_pairs": enriched_qa_pairs
    }

    filename = f"{paper_number}_{publication_year}_{short_id}.json"
    path = os.path.join(save_dir, filename)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    return path
