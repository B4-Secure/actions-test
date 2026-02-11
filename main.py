import os
import re
import glob
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import pandas as pd
import numpy as np
import requests
from dateutil import parser as dateparser
import trafilatura

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# CONFIG (can be overridden by GitHub Actions env vars)
# ---------------------------
PAST_DAYS = int(os.getenv("PAST_DAYS", "1"))
LOOKBACK_HOURS = int((os.getenv("LOOKBACK_HOURS") or "18").strip())
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "30"))
DUP_THRESHOLD = float(os.getenv("DUP_THRESHOLD", "0.60"))
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

# Feature flags for data sources
USE_NEWSAPI = os.getenv("USE_NEWSAPI", "false").lower() == "true"
USE_BING_NEWS = os.getenv("USE_BING_NEWS", "false").lower() == "true"
USE_GDELT = os.getenv("USE_GDELT", "true").lower() == "true"  # Enabled by default (free!)
EXTRACT_CONTENT = os.getenv("EXTRACT_CONTENT", "true").lower() == "true"  # Article extraction

# API Keys
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
BING_NEWS_KEY = os.getenv("BING_NEWS_KEY", "")

# Content extraction settings
MAX_EXTRACT_WORKERS = int(os.getenv("MAX_EXTRACT_WORKERS", "5"))  # Parallel extraction threads
EXTRACT_TIMEOUT = int(os.getenv("EXTRACT_TIMEOUT", "10"))  # Seconds per article

DEFAULT_HL, DEFAULT_GL, DEFAULT_CEID = "en-GB", "GB", "GB:en"


REGION_RULES = [
    # Belgium
    
    (r"Belgium.*English",("en-GB", "BE", "BE:en")),

    # Germany
   
    (r"Germany.*English",("en-GB", "DE", "DE:en")),

    # Spain
    
    (r"Spain.*English",  ("en-GB", "ES", "ES:en")),

    # France
    
    (r"France.*English", ("en-GB", "FR", "FR:en")),

    # Italy
   
    (r"Italy.*English",  ("en-GB", "IT", "IT:en")),
]


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------
# Your search library (unchanged)
# ---------------------------
SEARCH_LIBRARY_TEXT = r"""
Brand search Google News and X - PROTEST   ("Gucci" OR "Ralph Lauren" OR "Dior" OR "Armani" OR "Louis Vuitton" OR "Valentino" OR "Burberry" OR "Dolce Gabbana" OR "Polo Ralph Lauren" OR "Cartier" OR "Calvin Klein" OR "Tommy Hilfiger" OR "Timberland" OR "Prada" OR "Nike" OR "Balenciaga" OR "Canada Goose") AND protest
Brand search Google News and X - BOYCOTT   ("Gucci" OR "Ralph Lauren" OR "Dior" OR "Armani" OR "Louis Vuitton" OR "Valentino" OR "Burberry" OR "Dolce Gabbana" OR "Polo Ralph Lauren" OR "Cartier" OR "Calvin Klein" OR "Tommy Hilfiger" OR "Timberland" OR "Prada" OR "Nike" OR "Balenciaga" OR "Canada Goose") AND boycott
Cotswold Designer Outlet   ("Cotswold Designer Outlet")
Value Retail	"Value Retail" OR "Bicester Collection"
London Marylebone   ("London Marylebone" OR "Marylebone station") AND (incident OR disruption OR delay OR closure OR police OR protest OR bomb OR explosion OR stabbing OR shooting OR "suspicious package")
Roermond outlet   ("Roermond Outlet" OR Roermond) AND (incident OR protest OR boycott OR police OR closure OR evacuation OR bomb OR explosion OR stabbing OR shooting OR crime)
BV Logistics Companies   ("DHL" OR "UPS" OR "DPD" OR "FedEx" OR "Amazon" OR "Royal Mail" OR "Parcelforce" OR "EVRI") AND (delivery OR supply OR logistics)
High Level City searches Belgium Dutch language   ("Brussels" OR "Maastricht" OR "Antwerp") AND (bom OR bommelding OR explosie OR schieten OR steekincident OR "verdacht pakket")
High Level City searches Belgium English language   ("Brussels" OR "Maastricht" OR "Antwerp") AND (bomb OR explosion OR shooting OR stabbing OR "suspicious package" OR "bomb threat")
High Level City searches Belgium French language   ("Brussels" OR "Maastricht" OR "Antwerp") AND (bombe OR explosion OR "colis suspect" OR poignarder OR fusillade)
Local Town Searches Maasmechelen Dutch	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND (incident OR protest OR boycot OR bom OR bommelding OR explosie OR evacuatie OR afzetting OR politie OR politie-inzet OR steekpartij OR steekincident OR schietpartij OR schieten OR dreiging OR "verdacht pakket" OR verdachte)
Local Town Searches Maasmechelen French	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND (incident OR manifestation OR protestation OR bombe OR "alerte à la bombe" OR "menace de bombe" OR explosion OR évacuation OR "colis suspect" OR agression OR "attaque au couteau" OR fusillade OR poignarder OR police OR menace)
Local Town Searches Maasmechelen English	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND (incident OR protest OR boycott OR police OR "armed police" OR evacuation OR cordon OR closure OR disruption OR delay OR threat OR terror OR bomb OR explosion OR shooting OR stabbing OR "suspicious package" OR "bomb threat")
Village searches Maasmechelen   ("Maasmechelen Village" OR ("Maasmechelen" AND ("designer outlet" OR outlet OR retail)))
High Level City searches Germany English   ("Frankfurt" OR "Cologne" OR "Munich") AND (bomb OR explosion OR shooting OR stabbing OR "suspicious package")
High Level City searches Germany German   ("Frankfurt" OR "Cologne" OR "Munich") AND (bombe OR explosion OR schießen OR messer OR "verdächtiges paket")
Local Town Searches Wertheim English	("Wertheim" OR "Wurzburg" OR "Aschaffenburg") AND (incident OR protest OR boycott OR police OR "armed police" OR evacuation OR cordon OR closure OR disruption OR threat OR bomb OR explosion OR shooting OR stabbing OR "suspicious package" OR "bomb threat")
Local Town Searches Wertheim German	("Wertheim" OR "Wurzburg" OR "Aschaffenburg") AND (incident OR protest OR boykott OR polizei OR polizeieinsatz OR bombe OR bombendrohung OR sprengsatz OR explosion OR evakuierung OR absperrung OR verdächtig OR "verdächtiges paket" OR schießen OR schuss OR messer OR messerangriff OR großalarm OR großeinsatz)
Village search Wertheim   ("Wertheim Village" OR ("Wertheim" AND ("designer outlet" OR outlet OR retail)))
XR and JSO Broad search   ("Extinction Rebellion" OR "Just Stop Oil")
XR and JSO Village search   ("Extinction Rebellion" OR "Just Stop Oil") AND ("Ingolstadt" OR "Kildare" OR "Bicester" OR "Wertheim" OR "Fidenza" OR "Maasmechelen")
Shoplifting UK   "shoplifting" OR "retail crime" OR "theft from person" OR "pickpocket gang"
High Level City searches Spain English   ("Madrid" OR "Barcelona") AND (bomb OR explosion OR shooting OR stabbing OR "suspicious package")
High Level City searches Spain Spanish   ("Madrid" OR "Barcelona") AND (bomba OR explosión OR tiroteo OR puñalada OR "paquete sospechoso")
Local Town Searches Las Rozas & La Roca English   ("Las Rozas" OR "La Roca" OR "Mataro" OR "Badalona") AND (protest OR boycott OR bomb OR explosion OR shooting OR stabbing)
Local Town Searches Las Rozas & La Roca Spanish   ("Las Rozas" OR "La Roca" OR "Mataro" OR "Badalona") AND (protesta OR boicot OR bomba OR explosión OR tiroteo OR puñalada)
Village Search Las Rozas & La Roca   ("Las Rozas Village" OR "La Roca Village")
Local Town Searches Bicester & Kildare	("Bicester" OR "Kildare" OR "Newbridge") AND (protest OR boycott OR bomb OR explosion OR shooting OR stabbing)
Village Search Bicester & Kildare   ("Bicester Village" OR "Kildare Village")
High Level City searches UK & Ireland   ("London" OR "Oxford" OR "Dublin") AND (bomb OR explosion OR shooting OR stabbing OR "suspicious package")
High Level City searches France English	"Paris" AND (bomb OR explosion OR shooting OR stabbing)
High Level City searches France French	"Paris" AND (bombe OR explosion OR fusillade OR poignarder)
Local Town Searches La Vallée French   ("Serris" OR "Chessy" OR "Disneyland Paris") AND (manifestation OR grève OR bombe OR explosion OR poignarder)
Village Search La Vallée	("La Vallée Village" OR ("Serris" OR "Chessy") AND ("designer outlet" OR outlet OR retail))
High Level City searches Italy English   ("Milan" OR "Bologna") AND (bomb OR explosion OR shooting OR stabbing)
High Level City searches Italy Italian   ("Milan" OR "Bologna") AND (bomba OR esplosione OR sparatoria OR accoltellamento)
Local Town Searches Fidenza Italian   ("Fidenza" OR "Parma" OR "Piacenza" OR "Cremona") AND (protesta OR boicottaggio OR bomba OR esplosione)
Village Search Fidenza	("Fidenza Village" OR ("Fidenza" AND ("designer outlet" OR outlet OR retail)))
""".strip()


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def parse_published_dt(published_str: str):
    """Parse published date string to UTC datetime."""
    if not published_str:
        return None
    try:
        dt = dateparser.parse(published_str)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def filter_last_n_hours(df, hours: int):
    """Filter dataframe to only include articles from the last N hours."""
    if df.empty:
        return df
    
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    df = df.copy()
    df["published_dt_utc"] = df["published"].apply(parse_published_dt)
    df = df[df["published_dt_utc"].notna()]
    df = df[df["published_dt_utc"] >= cutoff].reset_index(drop=True)
    return df


def edition_for_search(search_name: str) -> tuple[str, str, str]:
    """
    Decide (hl, gl, ceid) for a given search based on search_name.
    Falls back to UK edition if nothing matches.
    """
    name = (search_name or "").strip()
    for pattern, triple in REGION_RULES:
        if re.search(pattern, name, flags=re.IGNORECASE):
            return triple
    return (DEFAULT_HL, DEFAULT_GL, DEFAULT_CEID)


def parse_search_library(text: str) -> pd.DataFrame:
    """Parse search library text into structured dataframe."""
    rows = []
    pending_name = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # URL on its own line
        if line.startswith("http"):
            if pending_name:
                rows.append({"search_name": pending_name, "raw_query": f'"{line}"'})
            else:
                rows.append({"search_name": "URL_SOURCE", "raw_query": f'"{line}"'})
            continue

        # Tab-separated "name<TAB>query"
        if "\t" in line:
            name, query = line.split("\t", 1)
            name = name.strip()
            query = query.strip()

            if query == "":
                pending_name = name
                continue

            if "\n" in query:
                parts = [p.strip().strip('"') for p in query.splitlines() if p.strip()]
                if parts and parts[0].startswith("http"):
                    for u in parts:
                        rows.append({"search_name": name, "raw_query": f'"{u}"'})
                    pending_name = None
                    continue

            rows.append({"search_name": name, "raw_query": query})
            pending_name = None
            continue

        # Fallback: split on 2+ spaces
        m = re.split(r"\s{2,}", line, maxsplit=1)
        if len(m) == 2:
            name, query = m[0].strip(), m[1].strip()
            rows.append({"search_name": name, "raw_query": query})
            pending_name = None
            continue

        # Can't parse
        rows.append({"search_name": "UNMAPPED_LINE", "raw_query": line})

    return pd.DataFrame(rows)


# ---------------------------
# NEWS COLLECTION FUNCTIONS
# ---------------------------

def create_fallback_query(query: str) -> str | None:
    """
    Create a simplified fallback query from a complex boolean query.
    
    For queries like: ("Maasmechelen" OR "Hasselt") AND (incident OR protest OR bomb)
    Returns: "Maasmechelen" OR "Hasselt"
    
    This helps get results for local searches where the full boolean is too restrictive.
    """
    # If query doesn't have AND, no need for fallback
    if ' AND ' not in query.upper():
        return None
    
    # Try to extract the first parenthetical group (usually location names)
    # Pattern: ("term1" OR "term2" OR ...) AND ...
    match = re.search(r'\(([^)]+)\)\s*AND', query, re.IGNORECASE)
    if match:
        location_part = match.group(1).strip()
        # Clean up and return just the locations
        return f"({location_part})"
    
    # Alternative: split on AND and take the first part
    parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
    if len(parts) >= 2:
        first_part = parts[0].strip()
        # Only use if it looks like it has location terms
        if first_part and len(first_part) > 3:
            return first_part
    
    return None


def google_news_rss_url(query: str, past_days: int, hl: str, gl: str, ceid: str) -> str:
    """Generate Google News RSS URL."""
    full = f"{query} when:{past_days}d"
    q = urllib.parse.quote(full)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


def fetch_google_news_rss(search_name: str, query: str, past_days: int, max_items: int) -> list[dict]:
    """Fetch articles from Google News RSS for a single search query."""
    hl, gl, ceid = edition_for_search(search_name)
    rss_url = google_news_rss_url(query, past_days, hl=hl, gl=gl, ceid=ceid)
    
    try:
        feed = feedparser.parse(rss_url)
        articles = []
        
        for entry in feed.entries[:max_items]:
            articles.append({
                "search_name": search_name,
                "search_query": query,
                "title": entry.get("title", ""),
                "published": entry.get("published", ""),
                "link": entry.get("link", ""),
                "source": "google_news_rss",
                "past_days": past_days,
                "hl": hl,
                "gl": gl,
                "ceid": ceid,
            })
        
        return articles
    except Exception as e:
        print(f"Error fetching RSS for '{search_name}': {e}")
        return []


def fetch_newsapi(search_name: str, query: str, hours_back: int, max_items: int = 100) -> list[dict]:
    """
    Fetch articles from NewsAPI (to be implemented).
    
    To enable:
    1. Get free API key from https://newsapi.org
    2. Set environment variable: NEWSAPI_KEY=your_key_here
    3. Set environment variable: USE_NEWSAPI=true
    """
    if not NEWSAPI_KEY:
        return []
    
    try:
        import requests
        
        from_date = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'publishedAt',
            'apiKey': NEWSAPI_KEY,
            'pageSize': min(max_items, 100),  # API limit
            'language': 'en'  # TODO: derive from region rules
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        articles = []
        
        for article in data.get('articles', []):
            articles.append({
                "search_name": search_name,
                "search_query": query,
                "title": article.get('title', ''),
                "published": article.get('publishedAt', ''),
                "link": article.get('url', ''),
                "source": f"newsapi_{article.get('source', {}).get('name', 'unknown')}",
                "description": article.get('description', ''),
            })
        
        return articles
        
    except Exception as e:
        print(f"Error fetching NewsAPI for '{search_name}': {e}")
        return []


def fetch_bing_news(search_name: str, query: str, hours_back: int, max_items: int = 100) -> list[dict]:
    """
    Fetch articles from Bing News Search API (to be implemented).
    
    To enable:
    1. Get API key from Azure Cognitive Services
    2. Set environment variable: BING_NEWS_KEY=your_key_here
    3. Set environment variable: USE_BING_NEWS=true
    """
    if not BING_NEWS_KEY:
        return []
    
    try:
        import requests
        
        url = "https://api.bing.microsoft.com/v7.0/news/search"
        headers = {"Ocp-Apim-Subscription-Key": BING_NEWS_KEY}
        params = {
            'q': query,
            'count': min(max_items, 100),
            'mkt': 'en-GB',  # TODO: derive from region rules
            'freshness': 'Day'  # Last 24 hours
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        articles = []
        
        for article in data.get('value', []):
            articles.append({
                "search_name": search_name,
                "search_query": query,
                "title": article.get('name', ''),
                "published": article.get('datePublished', ''),
                "link": article.get('url', ''),
                "source": f"bing_{article.get('provider', [{}])[0].get('name', 'unknown')}",
                "description": article.get('description', ''),
            })
        
        return articles
        
    except Exception as e:
        print(f"Error fetching Bing News for '{search_name}': {e}")
        return []


def fetch_gdelt(search_name: str, query: str, hours_back: int, max_items: int = 100) -> list[dict]:
    """
    Fetch articles from GDELT Project API (FREE, no API key needed!).
    
    GDELT monitors news from virtually every country in 100+ languages,
    updating every 15 minutes with ~300,000 articles daily.
    
    API Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    """
    try:
        # Clean up query for GDELT - it uses a simpler syntax
        # Remove boolean operators and quotes for basic keyword search
        clean_query = query
        clean_query = re.sub(r'\bAND\b', ' ', clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r'\bOR\b', ' ', clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r'["\(\)]', '', clean_query)
        clean_query = ' '.join(clean_query.split())  # Normalize whitespace
        
        # Take first few significant terms to avoid overly complex queries
        terms = [t.strip() for t in clean_query.split() if len(t.strip()) > 2][:5]
        gdelt_query = ' '.join(terms)
        
        if not gdelt_query:
            return []
        
        # GDELT DOC 2.0 API
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours_back)
        
        params = {
            'query': gdelt_query,
            'mode': 'artlist',  # Article list mode
            'maxrecords': min(max_items, 250),  # API limit is 250
            'format': 'json',
            'startdatetime': start_time.strftime('%Y%m%d%H%M%S'),
            'enddatetime': end_time.strftime('%Y%m%d%H%M%S'),
            'sort': 'datedesc'  # Most recent first
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        articles = []
        
        for article in data.get('articles', []):
            # Parse GDELT date format (YYYYMMDDHHMMSS)
            seendate = article.get('seendate', '')
            try:
                if seendate:
                    pub_dt = datetime.strptime(seendate, '%Y%m%d%H%M%S')
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    published = pub_dt.isoformat()
                else:
                    published = ''
            except:
                published = seendate
            
            articles.append({
                "search_name": search_name,
                "search_query": query,
                "title": article.get('title', ''),
                "published": published,
                "link": article.get('url', ''),
                "source": f"gdelt_{article.get('domain', 'unknown')}",
                "description": '',  # GDELT doesn't provide descriptions
                "language": article.get('language', ''),
                "source_country": article.get('sourcecountry', ''),
            })
        
        return articles
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching GDELT for '{search_name}': {e}")
        return []
    except Exception as e:
        print(f"Unexpected error fetching GDELT for '{search_name}': {e}")
        return []


def extract_article_content(url: str, timeout: int = 10) -> dict:
    """
    Extract full article content from a URL using trafilatura.
    
    Returns dict with: content, author, date, sitename
    """
    try:
        # Download the page
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {"content": None, "extraction_error": "Failed to download"}
        
        # Extract main content
        content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True
        )
        
        # Also try to get metadata
        metadata = trafilatura.extract_metadata(downloaded)
        
        return {
            "content": content,
            "author": metadata.author if metadata else None,
            "sitename": metadata.sitename if metadata else None,
            "extraction_error": None
        }
        
    except Exception as e:
        return {"content": None, "extraction_error": str(e)}


def extract_content_batch(df: pd.DataFrame, max_workers: int = 5) -> pd.DataFrame:
    """
    Extract article content for all URLs in the dataframe using parallel processing.
    """
    if df.empty or 'link' not in df.columns:
        return df
    
    print(f"\n{'='*60}")
    print(f"Extracting article content...")
    print(f"  - Articles to process: {len(df)}")
    print(f"  - Parallel workers: {max_workers}")
    print(f"{'='*60}\n")
    
    # Initialize new columns
    df = df.copy()
    df['content'] = None
    df['author'] = None
    df['sitename'] = None
    df['extraction_error'] = None
    
    urls = df['link'].tolist()
    results = {}
    
    start_time = time.time()
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all extraction tasks
        future_to_url = {
            executor.submit(extract_article_content, url, EXTRACT_TIMEOUT): url 
            for url in urls
        }
        
        # Process results as they complete
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            completed += 1
            
            try:
                result = future.result(timeout=EXTRACT_TIMEOUT + 5)
                results[url] = result
            except Exception as e:
                results[url] = {"content": None, "extraction_error": str(e)}
            
            # Progress update every 10 articles
            if completed % 10 == 0 or completed == len(urls):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  Progress: {completed}/{len(urls)} ({rate:.1f} articles/sec)")
    
    # Update dataframe with results
    for idx, row in df.iterrows():
        url = row['link']
        if url in results:
            result = results[url]
            df.at[idx, 'content'] = result.get('content')
            df.at[idx, 'author'] = result.get('author')
            df.at[idx, 'sitename'] = result.get('sitename')
            df.at[idx, 'extraction_error'] = result.get('extraction_error')
    
    # Summary stats
    success_count = df['content'].notna().sum()
    fail_count = df['content'].isna().sum()
    elapsed = time.time() - start_time
    
    print(f"\n✅ Content extraction complete:")
    print(f"  - Successful: {success_count} ({success_count/len(df)*100:.1f}%)")
    print(f"  - Failed: {fail_count}")
    print(f"  - Total time: {elapsed:.1f}s")
    
    return df


def collect_all_news(df_searches: pd.DataFrame, past_days: int, lookback_hours: int, max_items: int) -> pd.DataFrame:
    """
    Collect news from all enabled sources.
    
    This function orchestrates fetching from multiple sources:
    - Google News RSS (always enabled)
    - NewsAPI (if USE_NEWSAPI=true and NEWSAPI_KEY is set)
    - Bing News (if USE_BING_NEWS=true and BING_NEWS_KEY is set)
    - GDELT (if USE_GDELT=true, FREE - no API key needed!)
    
    Then optionally extracts full article content using trafilatura.
    """
    all_articles = []
    
    print(f"\n{'='*60}")
    print(f"Starting news collection:")
    print(f"  - Total searches: {len(df_searches)}")
    print(f"  - Google News RSS: ENABLED")
    print(f"  - NewsAPI: {'ENABLED' if USE_NEWSAPI and NEWSAPI_KEY else 'DISABLED'}")
    print(f"  - Bing News: {'ENABLED' if USE_BING_NEWS and BING_NEWS_KEY else 'DISABLED'}")
    print(f"  - GDELT: {'ENABLED' if USE_GDELT else 'DISABLED'}")
    print(f"  - Content Extraction: {'ENABLED' if EXTRACT_CONTENT else 'DISABLED'}")
    print(f"{'='*60}\n")
    
    fallback_count = 0  # Track how many fallback searches were used
    
    for idx, row in df_searches.iterrows():
        search_name = row["search_name"]
        query = row["raw_query"]
        
        print(f"[{idx+1}/{len(df_searches)}] Processing: {search_name[:50]}...")
        
        search_total = 0  # Track total articles for this search
        
        # Always try Google News RSS
        rss_articles = fetch_google_news_rss(search_name, query, past_days, max_items)
        all_articles.extend(rss_articles)
        search_total += len(rss_articles)
        print(f"  ├─ Google RSS: {len(rss_articles)} articles")
        
        # Try NewsAPI if enabled
        if USE_NEWSAPI and NEWSAPI_KEY:
            api_articles = fetch_newsapi(search_name, query, lookback_hours, max_items)
            all_articles.extend(api_articles)
            search_total += len(api_articles)
            print(f"  ├─ NewsAPI: {len(api_articles)} articles")
        
        # Try Bing News if enabled
        if USE_BING_NEWS and BING_NEWS_KEY:
            bing_articles = fetch_bing_news(search_name, query, lookback_hours, max_items)
            all_articles.extend(bing_articles)
            search_total += len(bing_articles)
            print(f"  ├─ Bing News: {len(bing_articles)} articles")
        
        # Try GDELT if enabled (FREE - no API key needed!)
        if USE_GDELT:
            gdelt_articles = fetch_gdelt(search_name, query, lookback_hours, max_items)
            all_articles.extend(gdelt_articles)
            search_total += len(gdelt_articles)
            print(f"  ├─ GDELT: {len(gdelt_articles)} articles")
        
        # FALLBACK: If no results from any source, try a relaxed query
        if search_total == 0:
            fallback_query = create_fallback_query(query)
            if fallback_query:
                print(f"  ├─ ⚠️  No results! Trying fallback query...")
                fallback_count += 1
                
                # Try fallback with Google RSS
                fallback_rss = fetch_google_news_rss(
                    f"{search_name} (fallback)", 
                    fallback_query, 
                    past_days, 
                    max_items
                )
                all_articles.extend(fallback_rss)
                
                # Try fallback with GDELT if enabled
                fallback_gdelt = []
                if USE_GDELT:
                    fallback_gdelt = fetch_gdelt(
                        f"{search_name} (fallback)", 
                        fallback_query, 
                        lookback_hours, 
                        max_items
                    )
                    all_articles.extend(fallback_gdelt)
                
                total_fallback = len(fallback_rss) + len(fallback_gdelt)
                print(f"  └─ 🔄 Fallback: {total_fallback} articles (query: {fallback_query[:50]}...)")
            else:
                print(f"  └─ ⚠️  No results and no fallback available")
        else:
            print(f"  └─ Total: {search_total} articles")
    
    if fallback_count > 0:
        print(f"\n📊 Used fallback searches for {fallback_count} queries with no results")
    
    df = pd.DataFrame(all_articles)
    
    if df.empty:
        print("\n⚠️  No articles collected from any source!")
        return df
    
    # Apply time filter
    print(f"\nBefore time filter: {len(df)} articles")
    df = filter_last_n_hours(df, hours=lookback_hours)
    print(f"After time filter ({lookback_hours}h): {len(df)} articles")
    
    # Remove exact URL duplicates (can happen when multiple sources return same article)
    df = df.drop_duplicates(subset=["link"]).reset_index(drop=True)
    print(f"After URL deduplication: {len(df)} articles")
    
    # Extract full article content if enabled
    if EXTRACT_CONTENT and not df.empty:
        df = extract_content_batch(df, max_workers=MAX_EXTRACT_WORKERS)
    
    return df


# ---------------------------
# DEDUPLICATION
# ---------------------------

def semantic_dedupe_csv(infile: str, out_clean: str, out_audit: str,
                        threshold: float, model_name: str) -> tuple[int, int]:
    """
    Deduplicate articles using semantic similarity.
    
    Returns: (original_count, cleaned_count)
    """
    print(f"\n{'='*60}")
    print(f"Starting semantic deduplication...")
    print(f"  - Input file: {infile}")
    print(f"  - Similarity threshold: {threshold}")
    print(f"  - Model: {model_name}")
    print(f"{'='*60}\n")
    
    df = pd.read_excel(infile)
    original_count = len(df)
    
    if df.empty:
        print("⚠️  Input file is empty, nothing to deduplicate")
        df.to_excel(out_clean, index=False, engine="openpyxl")
        pd.DataFrame().to_excel(out_audit, index=False, engine="openpyxl")
        return 0, 0
    
    df["compare_text"] = df["title"].fillna("").astype(str)
    
    # Filter out empty titles
    mask = df["compare_text"].str.len() > 0
    df_work = df[mask].copy().reset_index(drop=True)
    orig_idx = df.index[mask].to_numpy()
    
    if df_work.empty:
        print("⚠️  No valid titles to compare")
        df.drop(columns=["compare_text"], errors="ignore").to_excel(out_clean, index=False, engine="openpyxl")
        pd.DataFrame().to_excel(out_audit, index=False, engine="openpyxl")
        return len(df), len(df)
    
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(df_work)} articles...")
    emb = model.encode(
        df_work["compare_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    
    print("Computing similarity matrix...")
    sim = cosine_similarity(emb, emb)
    n = sim.shape[0]
    
    # Union-Find for grouping duplicates
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # Group similar articles
    duplicate_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)
                duplicate_pairs += 1

    print(f"Found {duplicate_pairs} similar pairs above threshold {threshold}")
    
    # Build groups
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    keep_work = set()
    audit_rows = []

    for g in groups.values():
        if len(g) == 1:
            keep_work.add(g[0])
            continue

        # Keep the earliest article (by original index)
        g_map = [(int(orig_idx[i]), i) for i in g]
        g_map.sort(key=lambda x: x[0])

        keep_orig, keep_i = g_map[0]
        keep_work.add(keep_i)

        for drop_orig, drop_i in g_map[1:]:
            audit_rows.append({
                "kept_original_row": keep_orig,
                "dropped_original_row": int(drop_orig),
                "similarity": float(sim[keep_i, drop_i]),
                "kept_title": df.loc[keep_orig, "title"],
                "dropped_title": df.loc[int(drop_orig), "title"],
            })

    kept_orig_rows = {int(orig_idx[i]) for i in keep_work}
    drop_orig_rows = set(map(int, orig_idx.tolist())) - kept_orig_rows

    keep_mask = np.ones(len(df), dtype=bool)
    for r in drop_orig_rows:
        keep_mask[r] = False

    df_clean = df.loc[keep_mask].drop(columns=["compare_text"], errors="ignore").reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)

    # Save results
    df_clean.to_excel(out_clean, index=False, engine="openpyxl")
    audit.to_excel(out_audit, index=False, engine="openpyxl")
    
    cleaned_count = len(df_clean)
    removed_count = original_count - cleaned_count
    
    print(f"\n✅ Deduplication complete:")
    print(f"  - Original articles: {original_count}")
    print(f"  - Removed duplicates: {removed_count}")
    print(f"  - Final articles: {cleaned_count}")
    print(f"  - Reduction: {(removed_count/original_count*100):.1f}%\n")
    
    return original_count, cleaned_count


# ---------------------------
# MAIN WORKFLOW
# ---------------------------

def main():
    """Main execution flow."""
    print("\n" + "="*60)
    print("NEWS COLLECTION AUTOMATION")
    print("="*60)
    
    ts = datetime.now(timezone.utc).strftime("%d_%m%y_UTC")
    
    # Parse search library
    print("\nParsing search library...")
    search_df = parse_search_library(SEARCH_LIBRARY_TEXT)
    
    # Filter out unmapped lines
    to_run = search_df[search_df["search_name"] != "UNMAPPED_LINE"].copy()
    skipped = search_df[search_df["search_name"] == "UNMAPPED_LINE"]
    
    print(f"✓ Parsed {len(to_run)} runnable searches")
    if not skipped.empty:
        print(f"⚠️  Skipped {len(skipped)} unmapped lines:")
        print(skipped.head(10).to_string(index=False))
    
    # Collect news from all sources
    results = collect_all_news(
        df_searches=to_run,
        past_days=PAST_DAYS,
        lookback_hours=LOOKBACK_HOURS,
        max_items=MAX_ITEMS
    )
    
    # Save raw results
    raw_results_file = DATA_DIR / f"google_news_raw_{ts}_past{PAST_DAYS}d.xlsx"
    audit_search_file = DATA_DIR / f"search_audit_{ts}.xlsx"
    
    if not results.empty:
        # Remove timezone info for Excel compatibility
        results = results.apply(
            lambda s: s.dt.tz_localize(None) 
            if hasattr(s, "dt") and getattr(s.dt, "tz", None) is not None 
            else s
        )
        
        results.to_excel(raw_results_file, index=False, engine="openpyxl")
        print(f"\n✓ Saved raw results: {raw_results_file} ({len(results)} articles)")
    else:
        print("\n⚠️  No results to save!")
        # Create empty file so workflow doesn't break
        pd.DataFrame().to_excel(raw_results_file, index=False, engine="openpyxl")
    
    search_df.to_excel(audit_search_file, index=False, engine="openpyxl")
    print(f"✓ Saved search audit: {audit_search_file}")
    
    # Semantic deduplication
    if not results.empty:
        dedup_file = DATA_DIR / f"google_news_dedup_{ts}_past{PAST_DAYS}d.xlsx"
        dedup_audit = DATA_DIR / f"google_news_dedup_audit_{ts}.xlsx"
        
        orig, cleaned = semantic_dedupe_csv(
            infile=str(raw_results_file),
            out_clean=str(dedup_file),
            out_audit=str(dedup_audit),
            threshold=DUP_THRESHOLD,
            model_name=MODEL_NAME,
        )
        
        # Always keep a stable file for automation
        latest = DATA_DIR / "latest_deduped.xlsx"
        shutil.copyfile(dedup_file, latest)
        
        print(f"✓ Saved deduplicated results: {dedup_file}")
        print(f"✓ Saved deduplication audit: {dedup_audit}")
        print(f"✓ Saved latest file: {latest}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  - Lookback hours: {LOOKBACK_HOURS}")
    print(f"  - Past days filter: {PAST_DAYS}")
    print(f"  - Max items per search: {MAX_ITEMS}")
    print(f"  - Dedup threshold: {DUP_THRESHOLD}")
    
    print(f"\nData Sources:")
    print(f"  - Google News RSS: ENABLED")
    print(f"  - NewsAPI: {'ENABLED' if USE_NEWSAPI and NEWSAPI_KEY else 'DISABLED'}")
    print(f"  - Bing News: {'ENABLED' if USE_BING_NEWS and BING_NEWS_KEY else 'DISABLED'}")
    print(f"  - GDELT: {'ENABLED' if USE_GDELT else 'DISABLED'}")
    print(f"  - Content Extraction: {'ENABLED' if EXTRACT_CONTENT else 'DISABLED'}")
    
    if not results.empty:
        print(f"\nTop searches by article count:")
        print(results["search_name"].value_counts().head(10).to_string())
        
        # Show source breakdown if multiple sources used
        if "source" in results.columns:
            print(f"\nArticles by source type:")
            source_counts = results["source"].apply(
                lambda x: x.split("_")[0] if pd.notna(x) and "_" in str(x) else "google_rss"
            ).value_counts()
            print(source_counts.to_string())
        
        # Content extraction stats
        if EXTRACT_CONTENT and "content" in results.columns:
            content_success = results["content"].notna().sum()
            print(f"\nContent extraction:")
            print(f"  - Successful: {content_success}/{len(results)} ({content_success/len(results)*100:.1f}%)")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()

