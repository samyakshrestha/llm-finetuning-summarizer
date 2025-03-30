import requests
import xml.etree.ElementTree as ET
import os

def search_arxiv(query="llm fine-tuning", max_results=25):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch from arXiv")

    root = ET.fromstring(response.content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    papers = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip()
        summary = entry.find('atom:summary', ns).text.strip()
        pdf_url = entry.find('atom:link[@title="pdf"]', ns).attrib['href']
        published = entry.find('atom:published', ns).text

        papers.append({
            "title": title,
            "summary": summary,
            "pdf_url": pdf_url,
            "published": published
        })

    return papers


def filter_papers(papers, keywords):
    filtered = []
    for paper in papers:
        content = (paper["title"] + " " + paper["summary"]).lower()
        if any(keyword.lower() in content for keyword in keywords):
            filtered.append(paper)
    return filtered


def download_papers(papers, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    for paper in papers:
        title_slug = paper["title"].replace(" ", "_").replace("/", "_")[:100]
        filename = os.path.join(download_dir, f"{title_slug}.pdf")

        if os.path.exists(filename):
            print(f"Already downloaded: {filename}")
            continue

        try:
            print(f"Downloading: {paper['title']}")
            pdf_response = requests.get(paper["pdf_url"])
            with open(filename, 'wb') as f:
                f.write(pdf_response.content)
        except Exception as e:
            print(f"Failed to download {paper['title']}: {e}")