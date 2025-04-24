import requests
import xml.etree.ElementTree as ET
import os
import time
import re
from datetime import datetime

# Configure logging if desired
import logging
logging.basicConfig(level=logging.INFO)


def search_arxiv(
    query: str,
    max_results: int = 100,
    start: int = 0,
    categories: list[str] = None
) -> list[dict]:
    """
    Query the arXiv API and return a list of papers (with metadata).

    Args:
        query: search string (e.g., "instruction tuning").
        max_results: number of results to fetch per call.
        start: offset for pagination.
        categories: optional arXiv categories to narrow search (e.g., ['cs.LG']).

    Returns:
        List of dicts with keys: arxiv_id, title, summary, pdf_url, published.
    """
    # Build category filter
    cat_query = ''
    if categories:
        cat_query = '+' + '+OR+'.join(f"cat:{c}" for c in categories)

    # Construct API URL
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=all:{query}{cat_query}"  
        f"&start={start}&max_results={max_results}"
    )
    logging.info(f"Querying arXiv: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    papers = []

    for entry in root.findall('atom:entry', ns):
        # Unique arXiv identifier
        full_id = entry.find('atom:id', ns).text
        arxiv_id = full_id.rsplit('/', 1)[-1]
        title = entry.find('atom:title', ns).text.strip()
        summary = entry.find('atom:summary', ns).text.strip()
        pdf_url = entry.find('atom:link[@title="pdf"]', ns).attrib['href']
        published = entry.find('atom:published', ns).text

        papers.append({
            'arxiv_id': arxiv_id,
            'title': title,
            'summary': summary,
            'pdf_url': pdf_url,
            'published': published
        })
    return papers


def filter_papers(
    papers: list[dict],
    keywords: list[str],
    year_from: int = 2021,
    top_k: int = 75
) -> list[dict]:
    """
    Filter, dedupe, and rank papers by keyword relevance and recency.

    Args:
        papers: list of papers from search_arxiv.
        keywords: list of keywords for relevance scoring.
        year_from: include papers published in or after this year.
        top_k: max number of papers to return.

    Returns:
        Top-k papers with added 'relevance_score', sorted desc.
    """
    seen_ids = set()
    scored = []

    for paper in papers:
        pid = paper['arxiv_id']
        # Deduplicate
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        # Check recency
        pub_year = datetime.fromisoformat(paper['published']).year
        if pub_year < year_from:
            continue

        # Keyword-based relevance
        text = (paper['title'] + ' ' + paper['summary']).lower()
        score = sum(text.count(k.lower()) for k in keywords)
        if score > 0:
            paper['relevance_score'] = score
            paper['pub_year'] = pub_year
            scored.append(paper)

    # Sort by relevance (desc), then recency (desc)
    scored.sort(key=lambda x: (-x['relevance_score'], -x['pub_year']))
    return scored[:top_k]


def download_papers(
    papers: list[dict],
    download_dir: str,
    sleep_time: float = 1.0
) -> None:
    """
    Download a list of paper PDFs to the specified directory, with throttling.

    Args:
        papers: list of dicts with 'arxiv_id' and 'pdf_url'.
        download_dir: path where PDFs will be saved.
        sleep_time: seconds to wait between downloads.
    """
    os.makedirs(download_dir, exist_ok=True)

    for paper in papers:
        pid = paper['arxiv_id']
        # Use arXiv ID for filename to avoid slug issues
        filename = os.path.join(download_dir, f"{pid}.pdf")

        if os.path.exists(filename):
            logging.info(f"Already downloaded: {filename}")
            continue

        try:
            logging.info(f"Downloading {pid}: {paper['title']}")
            resp = requests.get(paper['pdf_url'], timeout=60)
            resp.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(resp.content)
        except Exception as e:
            logging.warning(f"Failed to download {pid}: {e}")

        time.sleep(sleep_time)
