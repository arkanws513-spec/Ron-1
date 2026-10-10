"""Build a small, traceable Arabic starter corpus from public Arabic Wikipedia extracts.

This is a bootstrap corpus for a first from-scratch training experiment, not a claim of
production-grade language ability. It downloads plain-text article extracts and records
the source titles and URLs for reproducibility.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

TITLES = [
    "اللغة العربية", "مصر", "تاريخ مصر", "جغرافيا مصر", "القاهرة",
    "الرياضيات", "الفيزياء", "الكيمياء", "علم الأحياء", "علوم الحاسوب",
    "الذكاء الاصطناعي", "تعلم الآلة", "الشبكات العصبية الاصطناعية",
    "الفضاء", "النظام الشمسي", "الأرض", "القمر", "الشمس",
    "التاريخ", "الجغرافيا", "الفلسفة", "الأدب العربي", "النحو العربي",
    "القرآن", "الطاقة", "المناخ", "الإنترنت", "البرمجة",
    "الصحة", "التعليم", "الاقتصاد", "علم النفس", "البيئة",
    "الروبوت", "الهندسة", "المنطق", "الإحصاء", "الطب",
    "الحضارة الإسلامية", "اللغة", "الموسوعة",
]
API = "https://ar.wikipedia.org/w/api.php"

def fetch(title):
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "exsectionformat": "plain", "redirects": 1, "format": "json",
        "formatversion": 2, "titles": title,
    })
    req = urllib.request.Request(
        API + "?" + params,
        headers={"User-Agent": "Ron1NativeTraining/0.1 (educational open-corpus bootstrap)"},
    )
    with urllib.request.urlopen(req, timeout=25) as response:
        data = json.load(response)
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or pages[0].get("invalid"):
        return None
    page = pages[0]
    extract = page.get("extract", "").strip()
    if len(extract) < 300:
        return None
    return {
        "requested_title": title,
        "title": page.get("title", title),
        "page_id": page.get("pageid"),
        "url": "https://ar.wikipedia.org/?curid=" + str(page.get("pageid", "")),
        "characters": len(extract),
        "text": extract,
    }

def main():
    out = Path("data/seed-corpus")
    out.mkdir(parents=True, exist_ok=True)
    records, chunks = [], []
    for title in TITLES:
        try:
            item = fetch(title)
            if item:
                records.append({k: v for k, v in item.items() if k != "text"})
                chunks.append("عنوان: " + item["title"] + "\n" + item["text"])
                print(f"Fetched: {item['title']} ({item['characters']} chars)", flush=True)
            else:
                print(f"Skipped unavailable/short page: {title}", flush=True)
        except Exception as exc:
            print(f"Skipped {title}: {exc}", flush=True)
        time.sleep(0.15)
    corpus = "\n\n".join(chunks).strip()
    if len(corpus) < 20000:
        raise SystemExit(f"Corpus too small ({len(corpus)} chars); refusing to train.")
    (out / "arabic_wikipedia.txt").write_text(corpus, encoding="utf-8")
    manifest = {
        "name": "Ron-1 Arabic Wikipedia bootstrap corpus",
        "purpose": "Small first from-scratch training experiment; not a production corpus",
        "source": "Arabic Wikipedia, retrieved as plaintext extracts via MediaWiki API",
        "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "article_count": len(records),
        "character_count": len(corpus),
        "articles": records,
        "license_note": "Wikipedia text is generally available under CC BY-SA and/or GFDL; preserve attribution and verify individual page licensing before redistribution.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Corpus ready: {len(records)} articles, {len(corpus)} characters.", flush=True)

if __name__ == "__main__":
    main()
