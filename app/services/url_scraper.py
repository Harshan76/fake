import re
import urllib.parse
from typing import Dict, Any, Optional
import requests
from bs4 import BeautifulSoup

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 TruthLens/2.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Cache-Control": "max-age=0"
}

def validate_url(url: str) -> bool:
    """Checks whether the URL is syntactically valid and has an http/https scheme."""
    try:
        parsed = urllib.parse.urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False

def scrape_article_from_url(url: str) -> Dict[str, Any]:
    """
    Fetches article content from a news hyperlink.
    Extracts title, publisher, author, published date, and full article body.
    Handles bot-blocking, paywalls, and connection timeouts gracefully.
    """
    clean_url = url.strip()
    if not validate_url(clean_url):
        return {
            "success": False,
            "error": "The provided URL is invalid. Please supply a complete address starting with http:// or https://."
        }

    domain = urllib.parse.urlparse(clean_url).netloc.lower()

    try:
        session = requests.Session()
        res = session.get(clean_url, headers=REQUEST_HEADERS, timeout=8, allow_redirects=True)
        
        # Check status codes
        if res.status_code in (401, 403):
            return {
                "success": False,
                "error": f"Access denied (HTTP {res.status_code}) by {domain}. The website employs automated anti-scraping or a mandatory paywall. Please copy the article text manually and verify using the Text Input tab."
            }
        elif res.status_code == 404:
            return {
                "success": False,
                "error": f"Article not found (HTTP 404) at {clean_url}. The page may have been deleted, moved, or entered incorrectly."
            }
        elif res.status_code != 200:
            return {
                "success": False,
                "error": f"Unable to retrieve webpage (HTTP {res.status_code}) from {domain}."
            }

        html = res.text
        if not html or len(html.strip()) < 200:
            return {
                "success": False,
                "error": "The server returned an empty or insufficient HTML response."
            }

        soup = BeautifulSoup(html, "html.parser")

        # Remove scripts, styles, iframes, navigation, footers, and advertisements
        for tag in soup(["script", "style", "nav", "footer", "aside", "header", "noscript", "svg", "button"]):
            tag.decompose()

        # 1. Extract Title
        title = None
        # Check og:title
        og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # 2. Extract Publisher
        publisher = None
        og_site = soup.find("meta", property="og:site_name")
        if og_site and og_site.get("content"):
            publisher = og_site["content"].strip()
        else:
            publisher = domain.replace("www.", "").capitalize()

        # 3. Extract Author
        author = None
        author_meta = (
            soup.find("meta", attrs={"name": "author"}) or
            soup.find("meta", property="article:author") or
            soup.find("meta", attrs={"name": "byl"})
        )
        if author_meta and author_meta.get("content"):
            author = author_meta["content"].strip()

        # 4. Extract Publish Date
        pub_date = None
        date_meta = (
            soup.find("meta", property="article:published_time") or
            soup.find("meta", attrs={"name": "publish-date"}) or
            soup.find("meta", attrs={"name": "pubdate"})
        )
        if date_meta and date_meta.get("content"):
            pub_date = date_meta["content"].strip()

        # 5. Extract Body Text
        # Look for <article> tag or main content wrappers
        article_elem = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r'content|article-body|story-body|post-content', re.I))
        
        paragraphs = []
        if article_elem:
            p_tags = article_elem.find_all("p")
        else:
            p_tags = soup.find_all("p")

        for p in p_tags:
            text = p.get_text(strip=True)
            # Filter boilerplate sentences
            if len(text) > 30 and not re.search(r'cookie|subscribe|all rights reserved|terms of service|newsletter', text, re.I):
                paragraphs.append(text)

        full_body = "\n\n".join(paragraphs).strip()
        word_count = len(full_body.split())

        if word_count < 20:
            return {
                "success": False,
                "error": f"Extracted only {word_count} words from {domain}. The page content may be dynamically rendered via JavaScript or behind an interactive gate. Please copy and paste the text directly into TruthLens."
            }

        return {
            "success": True,
            "url": clean_url,
            "title": title or "Untitled News Article",
            "author": author,
            "publisher": publisher,
            "publish_date": pub_date,
            "extracted_text": full_body,
            "word_count": word_count
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Connection to {domain} timed out after 8 seconds. The remote server may be overloaded or blocking requests."
        }
    except requests.exceptions.SSLError:
        return {
            "success": False,
            "error": f"SSL certificate verification failed for {domain}."
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Failed to retrieve article from {domain}: {str(e)}"
        }
