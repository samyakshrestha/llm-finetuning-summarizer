import os
import json

def generate_QA_pair(paper_number, publication_year, short_id, title, qa_pairs, save_dir):
    data = {
        "paper_id": f"{paper_number}_{publication_year}_{short_id}",
        "title": title,
        "qa_pairs": qa_pairs
    }
    filename = f"{paper_number}_{publication_year}_{short_id}.json"
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    return path