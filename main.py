import os
import re
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
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# CONFIG
# ---------------------------
PAST_DAYS       = int(os.getenv("PAST_DAYS", "1"))      # ← 2 so RSS fetches wide, time filter trims precisely
LOOKBACK_HOURS  = int((os.getenv("LOOKBACK_HOURS") or "24").strip())
MAX_ITEMS       = int(os.getenv("MAX_ITEMS", "30"))
DUP_THRESHOLD   = float(os.getenv("DUP_THRESHOLD", "0.7"))
MODEL_NAME      = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
EXTRACT_CONTENT = os.getenv("EXTRACT_CONTENT", "false").lower() == "true"
TRANSLATE_TITLES = os.getenv("TRANSLATE_TITLES", "true").lower() == "true"
MAX_EXTRACT_WORKERS = int(os.getenv("MAX_EXTRACT_WORKERS", "5"))
EXTRACT_TIMEOUT     = int(os.getenv("EXTRACT_TIMEOUT", "15"))

DEFAULT_HL, DEFAULT_GL, DEFAULT_CEID = "en-GB", "GB", "GB:en"

REGION_RULES = [
    (r"Belgium.*English",     ("en-GB", "BE", "BE:en")),
    (r"Belgium.*Dutch",       ("nl",    "BE", "BE:nl")),
    (r"Belgium.*French",      ("fr",    "BE", "BE:fr")),
    (r"Germany.*English",     ("en-GB", "DE", "DE:en")),
    (r"Germany.*German",      ("de",    "DE", "DE:de")),
    (r"Spain.*English",       ("en-GB", "ES", "ES:en")),
    (r"Spain.*Spanish",       ("es",    "ES", "ES:es")),
    (r"France.*English",      ("en-GB", "FR", "FR:en")),
    (r"France.*French",       ("fr",    "FR", "FR:fr")),
    (r"Italy.*English",       ("en-GB", "IT", "IT:en")),
    (r"Italy.*Italian",       ("it",    "IT", "IT:it")),
    (r"Maasmechelen.*Dutch",  ("nl",    "BE", "BE:nl")),
    (r"Maasmechelen.*French", ("fr",    "BE", "BE:fr")),
    (r"Wertheim.*German",     ("de",    "DE", "DE:de")),
    (r"Ingolstadt.*German",   ("de",    "DE", "DE:de")),
    (r"Las Rozas.*Spanish",   ("es",    "ES", "ES:es")),
    (r"La Val.*French",       ("fr",    "FR", "FR:fr")),
    (r"Fidenza.*Italian",     ("it",    "IT", "IT:it")),
]

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------
# SEARCH LIBRARY  (v3 — tightened to reduce noise)
# ---------------------------
SEARCH_LIBRARY_TEXT = r"""
Brand Protest	("Gucci" OR "Ralph Lauren" OR "Dior" OR "Armani" OR "Louis Vuitton" OR "Burberry" OR "Prada" OR "Nike" OR "Canada Goose" OR "Hugo Boss" OR "Moncler" OR "Saint Laurent" OR "Loro Piana" OR "Lacoste" OR "Versace" OR "Jimmy Choo" OR "Michael Kors" OR "Missoni" OR "Kiton" OR "Aquazzura" OR "Tumi") AND ("protest" OR "animal rights" OR "fur free" OR "PETA" OR "activist") AND -BAFTA AND -"red carpet" AND -"awards" AND -"wears" AND -"dressed in"
Brand Boycott	("Gucci" OR "Ralph Lauren" OR "Dior" OR "Armani" OR "Louis Vuitton" OR "Burberry" OR "Prada" OR "Nike" OR "Canada Goose" OR "Hugo Boss" OR "Moncler" OR "Saint Laurent" OR "Loro Piana" OR "Lacoste" OR "Versace" OR "Jimmy Choo" OR "Michael Kors" OR "Missoni" OR "Kiton" OR "Aquazzura" OR "Tumi") AND ("boycott") AND -cricket AND -"travel boycott" AND -"chocolate"
Value Retail Brand	"Value Retail" OR "Bicester Collection" OR "Chic Outlet Shopping"
BV Value Retail Crime	("Bicester Village") AND ("money laundering" OR "gang" OR "criminal" OR "fraud" OR "counterfeit")
BV Logistics Disruption	("DHL" OR "UPS" OR "DPD" OR "FedEx" OR "Royal Mail" OR "Parcelforce" OR "EVRI" OR "Dropit") AND (disruption OR strike OR failure OR delay) AND ("UK" OR "United Kingdom" OR "Britain") AND -dental AND -"pay dispute" AND -"postal vote"
PETA Broad Search	"PETA" AND ("Italy" OR "UK" OR "Germany" OR "Belgium" OR "Spain" OR "France") AND ("fashion" OR "designer" OR "outlet" OR "luxury retail" OR "shopping village") AND (protest OR rally OR demonstration OR campaign OR action)
PETA Village Search	("PETA") AND ("Ingolstadt Village" OR "Kildare Village" OR "La Vallee Village" OR "Bicester Village" OR "Wertheim Village" OR "Las Rozas Village" OR "La Roca Village" OR "Fidenza Village" OR "Maasmechelen Village")
XR JSO Village Search	("Extinction Rebellion" OR "Just Stop Oil" OR "climate protest" OR "environmental protest") AND ("Ingolstadt" OR "Kildare" OR "Vallee" OR "Bicester" OR "Wertheim" OR "Las Rozas" OR "La Roca" OR "Fidenza" OR "Maasmechelen")
Shoplifting UK	("shoplifting" OR "retail theft" OR "retail crime" OR "shop theft" OR "pickpocket") AND ("UK" OR "England" OR "Britain") AND ("gang" OR "organised" OR "trend" OR "rise" OR "survey" OR "report")
London Marylebone Incident	("London Marylebone" OR "Marylebone station") AND (incident OR disruption OR closure OR police OR bomb OR explosion OR stabbing OR shooting OR "suspicious package")
High Level City Belgium Dutch	("Brussel" OR "Antwerpen" OR "Maastricht") AND (bom OR bommelding OR "verdacht pakket" OR "onbeheerd pakket" OR explosie OR schietpartij OR steekpartij OR liquidatie OR arrestatie)
High Level City Belgium English	("Brussels" OR "Antwerp" OR "Maastricht") AND (bomb OR explosion OR shooting OR stabbing OR "suspicious package" OR "bomb threat" OR "terror") AND -football AND -soccer AND -transfer
High Level City Belgium French	("Bruxelles" OR "Anvers" OR "Maastricht") AND (bombe OR explosion OR fusillade OR agression OR couteau OR "colis suspect" OR "alerte à la bombe" OR attentat) AND -football AND -transfert
Local Town Maasmechelen Dutch	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND (protest OR bom OR bommelding OR explosie OR schietpartij OR steekpartij OR evacuatie OR politie OR arrestatie) AND -sport AND -voetbal
Local Town Maasmechelen French	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND (manifestation OR bombe OR explosion OR fusillade OR couteau OR évacuation OR police OR arrestation OR protestation)
Local Town Maasmechelen English	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND (protest OR bomb OR explosion OR shooting OR stabbing OR evacuation OR police OR arrest OR threat) AND -sport AND -football
Village Maasmechelen	("Maasmechelen outlet" OR "Designer Outlet Maasmechelen" OR "Maasmechelen Village")
High Level City Germany English	("Frankfurt" OR "Cologne" OR "Munich") AND "Germany" AND (bomb OR explosion OR shooting OR stabbing OR "suspicious package" OR "bomb threat" OR "terror attack") AND -football AND -soccer AND -Bundesliga AND -Bayern AND -weather AND -snow AND -strike AND -"bomb cyclone"
High Level City Germany German	("Frankfurt" OR "Köln" OR "München") AND (Bombe OR Bombendrohung OR Explosion OR Schießerei OR Messerangriff OR Terroranschlag OR "verdächtiges Paket") AND -Fußball AND -Bundesliga AND -Bayern AND -Wetter
Local Town Wertheim English	("Wertheim outlet" OR "Designer Outlet Wertheim" OR "Wertheim Village" OR "Wertheim Germany") AND (protest OR bomb OR shooting OR explosion OR stabbing OR "suspicious package" OR evacuation OR police OR news)
Local Town Wertheim German	("Wertheim" OR "Würzburg" OR "Aschaffenburg") AND (Protest OR Bombe OR Bombendrohung OR Explosion OR Schießerei OR Polizei OR Evakuierung OR Terrorverdacht) AND -Wetter AND -Schnee
Village Wertheim	("Wertheim outlet" OR "Designer Outlet Wertheim" OR "Wertheim Village")
Local Town Ingolstadt English	("Ingolstadt outlet" OR "Designer Outlet Ingolstadt" OR "Ingolstadt Village") AND (protest OR bomb OR shooting OR explosion OR stabbing OR "suspicious package" OR evacuation OR police OR news)
Local Town Ingolstadt German	("Ingolstadt") AND (Protest OR Bombe OR Bombendrohung OR Explosion OR Schießerei OR Polizei OR Evakuierung OR Terrorverdacht) AND -Audi AND -Auto AND -Wetter
Village Ingolstadt	("Ingolstadt outlet" OR "Designer Outlet Ingolstadt" OR "Ingolstadt Village")
High Level City Spain English	("Madrid" OR "Barcelona") AND "Spain" AND ("bomb threat" OR "suspicious package" OR "stabbing attack" OR "shooting attack" OR "terror attack" OR "police operation" OR "evacuation") AND -football AND -soccer AND -transfer AND -signing AND -LaLiga AND -"Champions League" AND -"Real Madrid" AND -"Atletico"
High Level City Spain Spanish	("Madrid" OR "Barcelona") AND ("amenaza de bomba" OR "paquete sospechoso" OR "paquete explosivo" OR "ataque con cuchillo" OR "tiroteo" OR "atentado" OR "operación policial" OR "evacuación" OR "artefacto explosivo") AND -fútbol AND -fichaje AND -Liga AND -transferencia
Local Town Las Rozas English	("Las Rozas" OR "La Roca del Vallès" OR "Mataró" OR "Badalona") AND (protest OR bomb OR explosion OR shooting OR stabbing OR evacuation OR "police operation" OR arrest) AND -football AND -transfer
Local Town Las Rozas Spanish	("Las Rozas" OR "La Roca del Vallès" OR "Mataró" OR "Badalona") AND (protesta OR "amenaza de bomba" OR explosión OR tiroteo OR acuchillamiento OR evacuación OR "operación policial" OR detención) AND -fútbol AND -fichaje
Village Las Rozas La Roca	("Las Rozas outlet" OR "La Roca outlet" OR "Las Rozas Village" OR "La Roca Village")
Local Town Bicester Kildare	("Bicester" OR "Kildare" OR "Newbridge") AND (bomb OR explosion OR shooting OR stabbing OR "suspicious package" OR "bomb threat" OR "terror" OR "police operation") AND -"petrol bomb" AND -sport AND -GAA AND -football
Village Bicester Kildare	("Bicester Village" OR "Kildare Village" OR "Bicester outlet" OR "Kildare outlet")
High Level City UK Ireland	("London" OR "Dublin") AND ("bomb" OR "terror" OR "bomb threat" OR "suspicious package" OR "police operation" OR "counter terrorism" OR "stabbing attack" OR "shooting incident") AND -"bomb squad" AND -film AND -movie AND -filming AND -"New London" AND -Connecticut
High Level City France English	("Paris" OR "Île-de-France") AND "France" AND ("bomb threat" OR "suspicious package" OR "shooting" OR "stabbing" OR "terror attack" OR "police operation" OR evacuation) AND -Texas AND -"Paris, Texas" AND -film AND -movie AND -football AND -transfer
High Level City France French	("Paris" OR "Île-de-France") AND ("alerte à la bombe" OR "colis suspect" OR "menace à la bombe" OR fusillade OR "attentat" OR "attaque au couteau" OR "opération de police" OR évacuation) AND -immobilier AND -"prix immobilier" AND -foot AND -transfert AND -Texas
Local Town La Vallee French	("Serris" OR "Chessy" OR "Bailly-Romainvilliers" OR "Magny-le-Hongre" OR "Seine-et-Marne") AND (manifestation OR "alerte à la bombe" OR "colis suspect" OR bombe OR fusillade OR couteau OR protestation OR évacuation OR attentat OR "opération de police")
Local Town La Vallee English	("Serris" OR "Chessy" OR "Bailly-Romainvilliers" OR "Disneyland Paris" OR "Seine-et-Marne") AND (protest OR "bomb threat" OR "suspicious package" OR explosion OR shooting OR stabbing OR evacuation OR "police operation") AND -"theme park ride" AND -"park closure"
Village La Vallee	("La Vallee outlet" OR "Val d'Europe outlet" OR "La Vallée Village" OR "La Vallee Village")
High Level City Italy English	("Milan" OR "Bologna") AND "Italy" AND ("bomb threat" OR "suspicious package" OR "stabbing attack" OR "shooting" OR "terror attack" OR "police operation" OR explosion OR evacuation) AND -Olympics AND -biathlon AND -skiing AND -skating AND -Cortina AND -"Winter Games" AND -"Winter Olympics" AND -curling AND -hockey AND -football AND -transfer AND -Serie
High Level City Italy Italian	("Milano" OR "Bologna") AND ("minaccia bomba" OR "pacco sospetto" OR "pacco esplosivo" OR sparatoria OR accoltellamento OR "attentato" OR "operazione di polizia" OR esplosione OR evacuazione) AND -Olimpiadi AND -pattinaggio AND -sci AND -Cortina AND -calcio AND -Serie AND -mercato
High Level City Italy English	("Milan" OR "Milano" OR "Bologna") AND ("Italy" OR "Italian") AND (protest OR demonstration OR rally OR "civil unrest" OR strike OR blockade OR march) AND -football AND -transfer AND -Serie AND -"fashion week" AND -catwalk AND -runway
High Level City Italy Italian	("Milano" OR "Bologna") AND (protesta OR manifestazione OR corteo OR sciopero OR blocco OR presidio OR "sit-in") AND -calcio AND -moda AND -"settimana della moda" AND -Serie AND -mercato
Local Town Fidenza English	("Fidenza outlet" OR "Designer Outlet Fidenza" OR "Fidenza Village" OR "Parma" OR "Piacenza" OR "Cremona") AND (protest OR bomb OR explosion OR shooting OR stabbing OR "suspicious package" OR evacuation OR "police operation") AND -football AND -transfer
Local Town Fidenza Italian	("Fidenza" OR "Parma" OR "Piacenza" OR "Cremona") AND (protesta OR "minaccia bomba" OR esplosione OR sparatoria OR accoltellamento OR evacuazione OR "operazione di polizia" OR arresto) AND -calcio AND -Serie AND -mercato
Village Fidenza	("Fidenza outlet" OR "Designer Outlet Fidenza" OR "Fidenza Village")
Roermond Outlet	("Designer Outlet Roermond" OR "Roermond outlet" OR "Roermond shopping")
""".strip()


# ---------------------------
# VILLAGE FALLBACK MAP
# When a village search still returns 0, broaden to location name only.
# ---------------------------
VILLAGE_FALLBACK_MAP = {
    # Village outlet searches — fallback drops event terms, keeps outlet name only
    r"village maasmechelen":          '("Maasmechelen Village" OR "Maasmechelen") AND (shopping OR outlet OR luxury OR retail OR news)',
    r"village wertheim":              '("Wertheim Village" OR "Wertheim") AND (shopping OR outlet OR luxury OR retail OR news)',
    r"village ingolstadt":            '("Ingolstadt Village" OR "Ingolstadt") AND (shopping OR outlet OR luxury OR retail OR news)',
    r"village las rozas|la roca":     '("Las Rozas Village" OR "La Roca Village" OR "Las Rozas" OR "La Roca") AND (shopping OR outlet OR luxury OR retail OR news)',
    r"village bicester|kildare":      '("Bicester Village" OR "Kildare Village" OR "Bicester" OR "Kildare") AND (shopping OR outlet OR luxury OR retail OR news)',
    r"village la val":                '("La Vallee Village" OR "La Vallée Village" OR "Serris" OR "Chessy") AND (shopping OR outlet OR luxury OR retail OR news)',
    r"village fidenza":               '("Fidenza Village" OR "Fidenza") AND (shopping OR outlet OR luxury OR retail OR news)',
    # Local town English fallbacks — anchor to outlet name so results stay relevant
    r"local town wertheim english":     '("Wertheim outlet" OR "Designer Outlet Wertheim" OR "Wertheim Germany")',
    r"local town ingolstadt english":   '("Ingolstadt outlet" OR "Designer Outlet Ingolstadt")',
    r"local town las rozas english":    '("Las Rozas outlet" OR "La Roca outlet" OR "Las Rozas Village" OR "La Roca Village")',
    r"local town bicester kildare":     '("Bicester Village" OR "Kildare Village" OR "Bicester outlet" OR "Kildare outlet")',
    r"local town la vallee english":    '("La Vallee outlet" OR "Val d\'Europe" OR "Disneyland Paris") AND (news OR incident OR police OR closure)',
    r"local town fidenza english":      '("Fidenza outlet" OR "Designer Outlet Fidenza" OR "Parma") AND (news OR incident OR police)',
    # PETA — stays scoped to retail/luxury context
    r"peta broad search":               '"PETA" AND ("outlet" OR "luxury" OR "shopping" OR "fashion" OR "fur" OR "leather") AND (protest OR campaign OR action)',
    r"peta village search":             '"PETA" AND ("designer outlet" OR "outlet village" OR "luxury shopping") AND (protest OR campaign OR demonstration)',
    # Roermond
    r"roermond outlet":                 '("Designer Outlet Roermond" OR "Roermond outlet" OR "Roermond shopping")',
}

# ---------------------------
# HELPERS
# ---------------------------

def parse_published_dt(published_str: str):
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


def filter_last_n_hours(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    df = df.copy()
    df["published_dt_utc"] = df["published"].apply(parse_published_dt)
    df = df[df["published_dt_utc"].notna()]
    df = df[df["published_dt_utc"] >= cutoff].reset_index(drop=True)
    return df


def edition_for_search(search_name: str) -> tuple:
    name = (search_name or "").strip()
    for pattern, triple in REGION_RULES:
        if re.search(pattern, name, flags=re.IGNORECASE):
            return triple
    return (DEFAULT_HL, DEFAULT_GL, DEFAULT_CEID)


def parse_search_library(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            name, query = line.split("\t", 1)
            rows.append({"search_name": name.strip(), "raw_query": query.strip()})
        else:
            m = re.split(r"\s{2,}", line, maxsplit=1)
            if len(m) == 2:
                rows.append({"search_name": m[0].strip(), "raw_query": m[1].strip()})
            else:
                rows.append({"search_name": "UNMAPPED_LINE", "raw_query": line})
    return pd.DataFrame(rows)


# ---------------------------
# FALLBACK LOGIC
# ---------------------------

def create_fallback_query(search_name: str, query: str) -> str | None:
    name_lower = search_name.lower()
    for pattern, fb_query in VILLAGE_FALLBACK_MAP.items():
        if re.search(pattern, name_lower, flags=re.IGNORECASE):
            return fb_query

    if ' AND ' not in query.upper():
        return None

    match = re.search(r'\(([^)]+)\)\s*AND', query, re.IGNORECASE)
    if match:
        return f"({match.group(1).strip()})"

    parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
    if len(parts) >= 2 and len(parts[0].strip()) > 3:
        return parts[0].strip()

    return None


# ---------------------------
# GOOGLE NEWS RSS  (sole source)
# ---------------------------

def google_news_rss_url(query: str, past_days: int, hl: str, gl: str, ceid: str) -> str:
    full = f"{query} when:{past_days}d"
    q = urllib.parse.quote(full)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


def fetch_google_news_rss(search_name: str, query: str, past_days: int, max_items: int) -> list:
    hl, gl, ceid = edition_for_search(search_name)
    rss_url = google_news_rss_url(query, past_days, hl=hl, gl=gl, ceid=ceid)
    try:
        feed = feedparser.parse(rss_url)
        articles = []
        for entry in feed.entries[:max_items]:
            articles.append({
                "search_name":  search_name,
                "search_query": query,
                "title":        entry.get("title", ""),
                "published":    entry.get("published", ""),
                "link":         entry.get("link", ""),
                "source":       "google_rss",
                "hl": hl, "gl": gl, "ceid": ceid,
            })
        return articles
    except Exception as e:
        print(f"  ⚠️  RSS error for '{search_name}': {e}")
        return []


# ---------------------------
# CONTENT EXTRACTION  (optional, off by default)
# ---------------------------

def extract_article_content(url: str, timeout: int = 10) -> dict:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {"content": None, "extraction_error": "Failed to download"}
        content  = trafilatura.extract(downloaded, include_comments=False, include_tables=True,
                                       no_fallback=False, favor_precision=True)
        metadata = trafilatura.extract_metadata(downloaded)
        return {
            "content":          content,
            "author":           metadata.author   if metadata else None,
            "sitename":         metadata.sitename if metadata else None,
            "extraction_error": None,
        }
    except Exception as e:
        return {"content": None, "extraction_error": str(e)}


def extract_content_batch(df: pd.DataFrame, max_workers: int = 5) -> pd.DataFrame:
    if df.empty or 'link' not in df.columns:
        return df
    print(f"\n{'='*60}")
    print(f"Extracting article content  ({len(df)} articles, {max_workers} workers)")
    print(f"{'='*60}\n")
    df = df.copy()
    for col in ('content', 'author', 'sitename', 'extraction_error'):
        df[col] = None
    urls    = df['link'].tolist()
    results = {}
    start   = time.time()
    done    = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fmap = {executor.submit(extract_article_content, url, EXTRACT_TIMEOUT): url for url in urls}
        for future in as_completed(fmap):
            url  = fmap[future]
            done += 1
            try:
                results[url] = future.result(timeout=EXTRACT_TIMEOUT + 5)
            except Exception as e:
                results[url] = {"content": None, "extraction_error": str(e)}
            if done % 10 == 0 or done == len(urls):
                elapsed = time.time() - start
                print(f"  Progress: {done}/{len(urls)} ({done/elapsed:.1f} art/sec)")
    for idx, row in df.iterrows():
        r = results.get(row['link'], {})
        df.at[idx, 'content']          = r.get('content')
        df.at[idx, 'author']           = r.get('author')
        df.at[idx, 'sitename']         = r.get('sitename')
        df.at[idx, 'extraction_error'] = r.get('extraction_error')
    ok = df['content'].notna().sum()
    print(f"\n✅ Extraction: {ok}/{len(df)} successful ({ok/len(df)*100:.1f}%)")
    return df


def translate_titles_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'title_en' column with English translations.
    - English titles are copied as-is
    - Non-English titles are translated via Google Translate (free, no API key)
    - Falls back to original title if translation fails
    """
    if df.empty or "title" not in df.columns:
        return df

    print(f"\nTranslating {len(df)} titles to English...")
    translator = GoogleTranslator(source="auto", target="en")
    titles_en  = []
    translated = 0
    errors     = 0

    for title in df["title"].fillna("").astype(str):
        if not title.strip():
            titles_en.append("")
            continue
        try:
            lang = detect(title)
        except LangDetectException:
            lang = "en"

        if lang == "en":
            titles_en.append(title)
            continue

        try:
            result = translator.translate(title)
            titles_en.append(result if result else title)
            translated += 1
        except Exception:
            titles_en.append(title)   # fall back to original
            errors += 1

    df = df.copy()
    # Insert title_en right after title column
    title_idx = df.columns.get_loc("title")
    df.insert(title_idx + 1, "title_en", titles_en)

    print(f"✅ Translation: {translated} translated, {len(df)-translated-errors} already English, {errors} errors")
    return df


# ---------------------------
# MAIN COLLECTION
# ---------------------------

def collect_all_news(df_searches: pd.DataFrame, past_days: int, lookback_hours: int, max_items: int) -> pd.DataFrame:
    all_articles   = []
    fallback_count = 0

    print(f"\n{'='*60}")
    print(f"NEWS COLLECTION  —  Google News RSS only")
    print(f"  Searches: {len(df_searches)}  |  Window: {lookback_hours}h  |  PAST_DAYS: {past_days}  |  Max: {max_items}")
    print(f"{'='*60}\n")

    for idx, row in df_searches.iterrows():
        search_name = row["search_name"]
        query       = row["raw_query"]
        print(f"[{idx+1}/{len(df_searches)}] {search_name[:65]}")

        articles = fetch_google_news_rss(search_name, query, past_days, max_items)
        print(f"  ├─ Google RSS: {len(articles)} articles")

        if articles:
            all_articles.extend(articles)
            print(f"  └─ Total: {len(articles)}")
        else:
            fb_query = create_fallback_query(search_name, query)
            if fb_query:
                fallback_count += 1
                fb_name = f"{search_name} (fallback)"
                print(f"  ├─ ⚠️  0 results — fallback: {fb_query[:65]}...")
                fb_articles = fetch_google_news_rss(fb_name, fb_query, past_days, max_items)
                all_articles.extend(fb_articles)
                print(f"  └─ 🔄 Fallback: {len(fb_articles)} articles")
            else:
                print(f"  └─ ⚠️  0 results, no fallback available")

    if fallback_count:
        print(f"\n📊 Fallback used for {fallback_count} zero-result searches")

    df = pd.DataFrame(all_articles)
    if df.empty:
        print("\n⚠️  No articles collected!")
        return df

    print(f"\nBefore time filter:        {len(df)} articles")
    df = filter_last_n_hours(df, hours=lookback_hours)
    print(f"After time filter ({lookback_hours}h):  {len(df)} articles")
    df = df.drop_duplicates(subset=["link"]).reset_index(drop=True)
    print(f"After URL deduplication:   {len(df)} articles")

    if EXTRACT_CONTENT:
        df = extract_content_batch(df, max_workers=MAX_EXTRACT_WORKERS)

    if TRANSLATE_TITLES:
        df = translate_titles_batch(df)

    return df


# ---------------------------
# SEMANTIC DEDUPLICATION
# ---------------------------

def semantic_dedupe(infile: str, out_clean: str, out_audit: str, threshold: float, model_name: str) -> tuple:
    print(f"\n{'='*60}")
    print(f"Semantic deduplication  (threshold={threshold})")
    print(f"{'='*60}\n")

    df = pd.read_excel(infile)
    original_count = len(df)

    if df.empty:
        df.to_excel(out_clean, index=False, engine="openpyxl")
        pd.DataFrame().to_excel(out_audit, index=False, engine="openpyxl")
        return 0, 0

    df["compare_text"] = df["title"].fillna("").astype(str)
    mask     = df["compare_text"].str.len() > 0
    df_work  = df[mask].copy().reset_index(drop=True)
    orig_idx = df.index[mask].to_numpy()

    if df_work.empty:
        df.drop(columns=["compare_text"], errors="ignore").to_excel(out_clean, index=False, engine="openpyxl")
        pd.DataFrame().to_excel(out_audit, index=False, engine="openpyxl")
        return len(df), len(df)

    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"Generating embeddings for {len(df_work)} articles...")
    emb = model.encode(df_work["compare_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)

    print("Computing similarity matrix...")
    sim = cosine_similarity(emb, emb)
    n   = sim.shape[0]

    parent = list(range(n))
    rank   = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if rank[ra] < rank[rb]:   parent[ra] = rb
        elif rank[ra] > rank[rb]: parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)
                pairs += 1
    print(f"Found {pairs} similar pairs above threshold {threshold}")

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    keep_work  = set()
    audit_rows = []

    for g in groups.values():
        if len(g) == 1:
            keep_work.add(g[0])
            continue
        g_map = sorted([(int(orig_idx[i]), i) for i in g])
        keep_orig, keep_i = g_map[0]
        keep_work.add(keep_i)
        for drop_orig, drop_i in g_map[1:]:
            audit_rows.append({
                "kept_original_row":    keep_orig,
                "dropped_original_row": int(drop_orig),
                "similarity":           float(sim[keep_i, drop_i]),
                "kept_title":           df.loc[keep_orig, "title"],
                "dropped_title":        df.loc[int(drop_orig), "title"],
            })

    kept_rows = {int(orig_idx[i]) for i in keep_work}
    drop_rows = set(map(int, orig_idx.tolist())) - kept_rows
    keep_mask = np.ones(len(df), dtype=bool)
    for r in drop_rows:
        keep_mask[r] = False

    df_clean = df.loc[keep_mask].drop(columns=["compare_text"], errors="ignore").reset_index(drop=True)
    pd.DataFrame(audit_rows).to_excel(out_audit, index=False, engine="openpyxl")
    df_clean.to_excel(out_clean, index=False, engine="openpyxl")

    removed = original_count - len(df_clean)
    print(f"\n✅ Dedup: {original_count} → {len(df_clean)} ({removed} removed, {removed/original_count*100:.1f}% reduction)")
    return original_count, len(df_clean)


# ---------------------------
# MAIN
# ---------------------------

def main():
    print("\n" + "="*60)
    print("NEWS COLLECTION  —  Google News RSS")
    print("="*60)

    ts = datetime.now(timezone.utc).strftime("%d_%m%y_UTC")

    print("\nParsing search library...")
    search_df = parse_search_library(SEARCH_LIBRARY_TEXT)
    to_run    = search_df[search_df["search_name"] != "UNMAPPED_LINE"].copy().reset_index(drop=True)
    skipped   = search_df[search_df["search_name"] == "UNMAPPED_LINE"]

    print(f"✓ {len(to_run)} runnable searches")
    if not skipped.empty:
        print(f"⚠️  {len(skipped)} unmapped lines skipped")

    results = collect_all_news(
        df_searches=to_run,
        past_days=PAST_DAYS,
        lookback_hours=LOOKBACK_HOURS,
        max_items=MAX_ITEMS,
    )

    raw_file   = DATA_DIR / f"google_news_raw_{ts}_past{PAST_DAYS}d.xlsx"
    audit_file = DATA_DIR / f"search_audit_{ts}.xlsx"

    if not results.empty:
        results = results.apply(
            lambda s: s.dt.tz_localize(None)
            if hasattr(s, "dt") and getattr(s.dt, "tz", None) is not None else s
        )
        results.to_excel(raw_file, index=False, engine="openpyxl")
        print(f"\n✓ Raw results saved: {raw_file} ({len(results)} articles)")
    else:
        print("\n⚠️  No results to save!")
        pd.DataFrame().to_excel(raw_file, index=False, engine="openpyxl")

    search_df.to_excel(audit_file, index=False, engine="openpyxl")
    print(f"✓ Search audit saved: {audit_file}")

    if not results.empty:
        dedup_file  = DATA_DIR / f"google_news_dedup_{ts}_past{PAST_DAYS}d.xlsx"
        dedup_audit = DATA_DIR / f"google_news_dedup_audit_{ts}.xlsx"

        semantic_dedupe(
            infile=str(raw_file),
            out_clean=str(dedup_file),
            out_audit=str(dedup_audit),
            threshold=DUP_THRESHOLD,
            model_name=MODEL_NAME,
        )

        latest = DATA_DIR / "latest_deduped.xlsx"
        shutil.copyfile(dedup_file, latest)
        print(f"✓ Deduplicated file: {dedup_file}")
        print(f"✓ Latest file:       {latest}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Lookback window  : {LOOKBACK_HOURS}h")
    print(f"  Past days filter : {PAST_DAYS}")
    print(f"  Max per search   : {MAX_ITEMS}")
    print(f"  Dedup threshold  : {DUP_THRESHOLD}")
    print(f"  Content extract  : {'ENABLED' if EXTRACT_CONTENT else 'DISABLED'}")

    if not results.empty:
        print(f"\nTop searches by article count:")
        print(results["search_name"].value_counts().head(15).to_string())

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
