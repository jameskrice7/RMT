#!/usr/bin/env python
"""Audit paper/references.bib against Crossref.

For entries with a DOI the Crossref record is fetched and title, year, volume
and first page are compared.  Entries without a DOI are looked up by
bibliographic search and the best hit is reported for manual inspection.
Writes reproduction/results/bibliography_audit.csv.
"""
import csv
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "references.bib"
UA = {"User-Agent": "rmt-replication-bib-audit/1.0"}


def parse_bib(text):
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\}\s*(?=\n@|\n%|\Z)", text, re.S):
        typ, key, body = m.group(1), m.group(2), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", body):
            fields[fm.group(1).lower()] = fm.group(2)
        entries.append((typ.lower(), key.strip(), fields))
    return entries


def clean(s):
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s or "")
    s = re.sub(r"[{}\\$^'`\"~]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    entries = parse_bib(BIB.read_text(encoding="utf8"))
    rows = []
    for typ, key, f in entries:
        row = dict(key=key, type=typ, doi=f.get("doi", ""), status="", note="")
        try:
            if f.get("doi"):
                msg = get("https://api.crossref.org/works/" + urllib.parse.quote(f["doi"]))["message"]
            elif typ in ("article", "inproceedings", "incollection", "book"):
                q = urllib.parse.quote(clean(f.get("title", "")) + " " + clean(f.get("author", f.get("editor", "")))[:80])
                items = get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=1")["message"]["items"]
                msg = items[0] if items else {}
                row["note"] = "search hit doi=" + msg.get("DOI", "")
            else:
                row["status"] = "skipped"
                rows.append(row)
                continue
            ct = clean((msg.get("title") or [""])[0])
            sim = difflib.SequenceMatcher(None, ct, clean(f.get("title", ""))).ratio()
            parts = (msg.get("issued") or {}).get("date-parts") or [[None]]
            year = str(parts[0][0])
            vol = msg.get("volume", "")
            page = (msg.get("page") or "").split("-")[0]
            issues = []
            if sim < 0.85:
                issues.append(f"title sim {sim:.2f}: '{ct[:70]}'")
            if f.get("year") and year != "None" and year != f["year"]:
                issues.append(f"year {year}")
            if f.get("volume") and vol and vol != f["volume"]:
                issues.append(f"volume {vol}")
            bp = (f.get("pages") or "").split("-")[0]
            if bp and page and page != bp:
                issues.append(f"page {page}")
            row["status"] = "ok" if not issues else "check"
            row["note"] = (row["note"] + " " + "; ".join(issues)).strip()
        except Exception as e:  # network or missing record
            row["status"] = "error"
            row["note"] = str(e)[:120]
        rows.append(row)
        print(f"{row['status']:7s} {key:32s} {row['note']}")
        sys.stdout.flush()
        time.sleep(0.15)
    out = ROOT / "reproduction" / "results" / "bibliography_audit.csv"
    with open(out, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
