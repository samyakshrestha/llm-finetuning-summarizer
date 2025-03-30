import requests # This library is used to make HTTP requests
import xml.etree.ElementTree as ET # This is a standard Python library for parsing XML.

# query: A string to search for. It defaults to "llm fine-tuning"
# max_results: The maximum number of results to fetch (default is 25)
def search_arxiv(query="llm fine-tuning", max_results=25):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch from arXiv")
    
    # parses the XML content of the response into an ElementTree
    root = ET.fromstring(response.content) 
    # helps in querying XML elements with the proper namespace
    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    papers = []
    # iterate over each <entry> element in the XML
    for entry in root.findall('atom:entry', ns):
        paper = {
            "title": entry.find('atom:title', ns).text.strip(),
            "summary": entry.find('atom:summary', ns).text.strip(),
            "pdf_url": entry.find('atom:link[@title="pdf"]', ns).attrib['href'],
            "published": entry.find('atom:published', ns).text
        }
        papers.append(paper)

    return papers
    # the function returns the list of papers containing all the metadata we extracted