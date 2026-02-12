"""
Extract ADHICS controls from the Abu Dhabi Healthcare Information and
Cyber Security Standard (ADHIC) PDF. Regulation policies are always PDF;
extraction is from PDF only. ADHICS uses domains, sub-domains, and
controls with Control Demand, Control Criteria, and Applicability (Basic/
Transitional/Advanced). The standard uses tables for controls in Section B.

Output format (ADHICS schema):
  standard_name, version, controls: [ { domain, sub_domain, control_id,
  control_demand, applicability, criteria[], subcontrols[], references[], source_index } ]
  - criteria: list of { "text": "...", "sub": [ "..." ] } (numbered 1. 2. 3. with lettered a. b. c. under each).
  - subcontrols: flat list of all lettered sub-items (a., b., c.) from criteria[].sub for easy access.

Usage:
  python3 extract_adhic_controls.py --pdf data/01_raw/regulation/ADHIC.pdf
  # Writes data/02_processed/adhics_controls_structured.json
  # Use --update-control-ids to refresh adhics_control_ids.txt
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import pdfplumber
    from pdfplumber.utils.exceptions import PdfminerException
except ImportError:
    pdfplumber = None
    PdfminerException = Exception  # type: ignore

# Domain number to name (Section B style) and clean display names
DOMAIN_NAMES = {
    1: "1. HUMAN RESOURCES SECURITY",
    2: "2. ASSET MANAGEMENT",
    3: "3. PHYSICAL AND ENVIRONMENTAL SECURITY",
    4: "4. ACCESS CONTROL",
    5: "5. OPERATIONS MANAGEMENT",
    6: "6. COMMUNICATIONS",
    7: "7. HEALTH INFORMATION AND SECURITY",
}
DOMAIN_NAMES_CLEAN = {
    1: "Human Resources Security",
    2: "Asset Management",
    3: "Physical and Environmental Security",
    4: "Access Control",
    5: "Operations Management",
    6: "Communications",
    7: "Health Information and Security",
}

# Sub-domain header pattern: "HR 1 ...", "AM 1 ...", "AC 2 ..."
SUB_DOMAIN_PATTERN = re.compile(
    r"^(HR|AM|PE|AC|OM|CO|HI)\s+(\d+)\s+(.+)$",
    re.IGNORECASE,
)
# Prefix to domain number (for correct domain when section order varies)
PREFIX_TO_DOMAIN = {"HR": 1, "AM": 2, "PE": 3, "AC": 4, "OM": 5, "CO": 6, "HI": 7}
# Control ID in text: "HR 1.1", "AM 3.6"
CONTROL_ID_PATTERN = re.compile(
    r"^(HR|AM|PE|AC|OM|CO|HI)\s+(\d+\.\d+)(?:\s|$|\[)",
    re.IGNORECASE,
)
# Control heading line: "AC 1.1 Access Control Policy" or "PE 2.4 Ownership [T]"
CONTROL_HEADING_PATTERN = re.compile(
    r"^(HR|AM|PE|AC|OM|CO|HI)\s+(\d+\.\d+)\s+(.+)$",
    re.IGNORECASE,
)
# Applicability in brackets: [B], [T], [A] -> Basic, Transitional, Advanced
APPLICABILITY_BRACKET = re.compile(r"\[([BTA])\]", re.IGNORECASE)
BRACKET_TO_APPLICABILITY = {"B": "Basic", "T": "Transitional", "A": "Advanced"}
# Domain line: "Domain 4 - Access Control"
DOMAIN_HEADING_PATTERN = re.compile(
    r"^Domain\s+(\d+)\s*[-–]\s*(.+)$",
    re.IGNORECASE,
)
# Applicability
APPLICABILITY_PATTERN = re.compile(
    r"(?:Applicability|Level)\s*[:\s]*(Basic|Transitional|Advanced)",
    re.IGNORECASE,
)
# UAE IA Reference(s)
UAE_REF_PATTERN = re.compile(
    r"UAE\s+IA\s+References?\s*:?\s*([^\n]+)",
    re.IGNORECASE,
)
# Split refs like "M3.1.1, M4.1.1" or "T1.2.1, T1.2.2 & T1.2.4"
def _parse_references(s: str) -> List[str]:
    s = re.sub(r"\s+&\s+", ", ", s)
    return [r.strip() for r in re.split(r"[,;]", s) if re.match(r"^[MT]\d+\.\d+", r.strip())]


def _clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"Page\s+\d+\s+of\s+\d+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"DEPARTMENT\s+OF\s+HEALTH", "", t, flags=re.IGNORECASE)
    return t.strip()


def _table_to_rows(table: List[List[Any]]) -> List[List[str]]:
    return [[str(c).strip() if c else "" for c in row] for row in table]


def _subcontrols_from_criteria(criteria: List[Dict[str, Any]]) -> List[str]:
    """Flatten criteria[].sub into a single list (all lettered sub-items)."""
    out: List[str] = []
    for item in criteria or []:
        sub = item.get("sub") or []
        out.extend(s for s in sub if (s or "").strip())
    return out


def _is_valid_control_id(s: str) -> bool:
    """True only for ADHICS control IDs like HR 1.1, AC 2.3 (no body text)."""
    if not s or len(s) > 20:
        return False
    return CONTROL_ID_PATTERN.match(s.strip()) is not None


def _domain_from_control_id(control_id: str) -> str:
    """Derive domain from control ID prefix (HR, AM, PE, ...) so domain is never from wrong page context."""
    prefix = (control_id or "").split()[0]
    num = PREFIX_TO_DOMAIN.get((prefix or "").upper())
    return DOMAIN_NAMES_CLEAN.get(num, "ADHICS") if num is not None else "ADHICS"


def _sub_domain_label(raw: str) -> str:
    """From 'HR 1 Human Resources Security Policy' return 'Human Resources Security Policy'."""
    m = SUB_DOMAIN_PATTERN.match((raw or "").strip())
    return m.group(3).strip() if m else (raw or "").strip()


class ADHICControlExtractor:
    """Extract ADHICS controls from PDF (tables) or parsed JSON."""

    def __init__(self):
        self.standard_name = "Abu Dhabi Healthcare Information and Cyber Security Standard (ADHICS)"
        self.version = "0.9"

    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract controls from ADHIC PDF using tables and surrounding text."""
        if not pdfplumber:
            raise ImportError("pip install pdfplumber")
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"Extracting ADHICS controls from PDF: {pdf_path}")
        controls = []
        current_domain = ""
        current_sub_domain = ""
        current_refs: List[str] = []

        try:
            pdf = pdfplumber.open(pdf_path)
        except PdfminerException as e:
            msg = (
                "The PDF could not be opened (possible corrupted or truncated file). "
                "Error: Unexpected EOF / No valid xref. "
                "Please ensure the file is complete: re-download the PDF or open and re-save it in a PDF viewer."
            )
            raise RuntimeError(msg) from e

        with pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                if page_num % 10 == 0:
                    print(f"  Page {page_num}/{len(pdf.pages)}...", end="\r")

                # Text for domain/sub-domain context
                text = page.extract_text() or ""
                text_clean = _clean_text(text)
                lines = [l.strip() for l in text_clean.split("\n") if l.strip()]

                for line in lines:
                    # Domain: "1. Human Resources Security" or "Domain 4 - Access Control"
                    dm = DOMAIN_HEADING_PATTERN.match(line)
                    if dm:
                        num = int(dm.group(1))
                        current_domain = DOMAIN_NAMES.get(num, f"{num}. {dm.group(2).strip().upper()}")
                    else:
                        m = re.match(r"^(\d+)\.\s+(.+)$", line)
                        if m and len(m.group(1)) <= 2:
                            num = m.group(1)
                            current_domain = DOMAIN_NAMES.get(int(num), f"{num}. {m.group(2).upper()}")

                    # Sub-domain: "HR 1 Human Resources Security Policy"
                    sm = SUB_DOMAIN_PATTERN.match(line)
                    if sm:
                        current_sub_domain = line
                        current_refs = []

                    # UAE IA Reference(s)
                    ref_m = UAE_REF_PATTERN.match(line)
                    if ref_m:
                        current_refs = _parse_references(ref_m.group(1))

                    # Applicability in line
                    app_m = APPLICABILITY_PATTERN.search(line)
                    applicability = app_m.group(1) if app_m else "Basic"

                # Process all tables on the page (multiple tables per page are supported)
                tables = page.extract_tables() or []
                for table in tables:
                    rows = _table_to_rows(table)
                    if len(rows) < 2:
                        continue
                    header = [h.upper() for h in rows[0]]
                    # Find column indices by common header names
                    id_col = _find_col(header, ["CONTROL ID", "CONTROL IDENTIFIER", "ID"])
                    demand_col = _find_col(header, ["CONTROL DEMAND", "DEMAND", "REQUIREMENT"])
                    app_col = _find_col(header, ["APPLICABILITY", "LEVEL"])
                    crit_col = _find_col(header, ["CRITERIA", "CONTROL CRITERIA"])
                    ref_col = _find_col(header, ["REFERENCE", "REFERENCES", "UAE IA"])

                    if id_col is None and demand_col is None:
                        continue

                    for row in rows[1:]:
                        control_id = (row[id_col] if id_col is not None and id_col < len(row) else "").strip()
                        control_demand = (row[demand_col] if demand_col is not None and demand_col < len(row) else "").strip()
                        app_val = (row[app_col] if app_col is not None and app_col < len(row) else "").strip()
                        criteria_text = (row[crit_col] if crit_col is not None and crit_col < len(row) else "").strip()
                        ref_text = (row[ref_col] if ref_col is not None and ref_col < len(row) else "").strip()

                        if not control_id and not control_demand:
                            continue
                        # Normalize control_id (e.g. "HR 1.1") from demand if needed
                        if not control_id and control_demand:
                            cid_m = CONTROL_ID_PATTERN.match(control_demand)
                            if cid_m:
                                control_id = f"{cid_m.group(1).upper()} {cid_m.group(2)}"
                        # Only accept rows with valid ADHICS control IDs (skip garbage tables)
                        if not control_id or not _is_valid_control_id(control_id):
                            continue
                        applicability_val = app_val or applicability
                        if applicability_val not in ("Basic", "Transitional", "Advanced"):
                            applicability_val = "Basic"
                        criteria_list_raw = _split_criteria(criteria_text)
                        # Support nested criteria: list of { "text", "sub" } if we detect structure
                        criteria_list: List[Dict[str, Any]] = []
                        if criteria_list_raw:
                            for item in criteria_list_raw:
                                criteria_list.append({"text": item, "sub": []})
                        refs = _parse_references(ref_text) if ref_text else list(current_refs)
                        controls.append({
                            "domain": _domain_from_control_id(control_id),
                            "sub_domain": _sub_domain_label(current_sub_domain) if current_sub_domain else "",
                            "control_id": control_id,
                            "control_demand": control_demand or "",
                            "applicability": applicability_val,
                            "criteria": criteria_list,
                            # "subcontrols": _subcontrols_from_criteria(criteria_list),
                            "references": refs,
                            "source_index": [],
                        })

        # Always run text-based extraction for full descriptions and nested criteria
        text_controls = self._extract_from_text_fallback(pdf_path)
        # Prefer text entry with longer control_demand and more criteria (detailed section over summary)
        by_id: Dict[str, Dict] = {}
        for c in text_controls:
            cid = c.get("control_id")
            if not cid:
                continue
            existing = by_id.get(cid)
            c_demand = c.get("control_demand") or ""
            c_crit = len(c.get("criteria") or [])
            if existing is None:
                by_id[cid] = c
            else:
                e_demand = existing.get("control_demand") or ""
                e_crit = len(existing.get("criteria") or [])
                if len(c_demand) > len(e_demand) or (len(c_demand) >= len(e_demand) * 0.8 and c_crit > e_crit):
                    by_id[cid] = c
        # Add table-only controls; enrich refs/applicability from table when we have both
        for c in controls:
            cid = c.get("control_id") or ""
            if not cid:
                continue
            if cid not in by_id:
                by_id[cid] = c
            else:
                existing = by_id[cid]
                if not (existing.get("references")):
                    existing["references"] = c.get("references") or []
                if not (existing.get("applicability") and c.get("applicability")):
                    existing["applicability"] = c.get("applicability") or existing.get("applicability", "Basic")
        controls = list(by_id.values())

        print(f"\n✓ Extracted {len(controls)} ADHICS controls from PDF")
        return self._build_output(controls)

    def _col_index(self, header: List[str], names: List[str]) -> Optional[int]:
        for n in names:
            for i, h in enumerate(header):
                if n in h or h in n:
                    return i
        return None

    def _extract_from_text_fallback(self, pdf_path: str) -> List[Dict]:
        """Extract controls from guideline text: headings like 'AC 1.1 ...' / 'PE 2.4 ... [T]' and following paragraphs."""
        controls = []
        current_domain = ""
        current_sub_domain = ""
        current_refs: List[str] = []
        # Current control being built (from last CONTROL_HEADING line)
        pending_id: Optional[str] = None
        pending_title: str = ""
        pending_applicability: str = "Basic"
        demand_lines: List[str] = []
        max_demand_lines = 50  # cap paragraph lines per control

        def flush_control():
            nonlocal pending_id, pending_title, pending_applicability, demand_lines, pending_criteria_from_page
            if not pending_id or not _is_valid_control_id(pending_id):
                demand_lines = []
                return
            # Domain from control ID prefix (not page context); sub_domain = descriptive label only
            domain = _domain_from_control_id(pending_id)
            sub_domain = _sub_domain_label(current_sub_domain) if current_sub_domain else current_sub_domain or ""
            crit_block = pending_criteria_from_page
            pending_criteria_from_page = []
            if crit_block:
                criteria_list = _parse_nested_criteria("\n".join(crit_block))
                description, _ = _parse_description_and_criteria(demand_lines, pending_title)
            else:
                description, criteria_list = _parse_description_and_criteria(demand_lines, pending_title)
            if not description:
                description = " ".join(demand_lines).strip()[:2000] if demand_lines else pending_title or pending_id
            controls.append({
                "domain": domain,
                "sub_domain": sub_domain,
                "control_id": pending_id,
                "control_demand": description[:2000] if len(description) > 2000 else description,
                "applicability": pending_applicability,
                "criteria": criteria_list,
                # "subcontrols": _subcontrols_from_criteria(criteria_list),
                "references": list(current_refs),
                "source_index": [],
            })
            pending_id = None
            pending_title = ""
            pending_applicability = "Basic"
            demand_lines = []

        def _criteria_block_from_page_lines(lines_before: List[str]) -> List[str]:
            """Find a 'The policy shall:' / '1.' block in lines before the control on the same page (PDF often has criteria column before demand column)."""
            out: List[str] = []
            start = -1
            for i, L in enumerate(lines_before):
                if re.match(r"^The policy shall\s*:?\s*$", L, re.I) or re.match(r"^\s*1\.\s+", L) or re.match(r"^\s*1[\.\)]\s*$", L):
                    start = i
                    break
            if start < 0:
                return out
            for i in range(start, len(lines_before)):
                L = lines_before[i]
                if CONTROL_HEADING_PATTERN.match(L) or SUB_DOMAIN_PATTERN.match(L):
                    break
                if L.startswith("UAE IA Reference") or L.startswith("Page "):
                    break
                out.append(L)
            return out

        pending_criteria_from_page: List[str] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                page_lines = [_clean_text(l) for l in text.split("\n") if _clean_text(l)]
                for idx, line in enumerate(page_lines):
                    # Domain: "Domain 4 - Access Control" or "4. Access Control"
                    dm = DOMAIN_HEADING_PATTERN.match(line)
                    if dm:
                        num = int(dm.group(1))
                        current_domain = DOMAIN_NAMES.get(num, f"{num}. {dm.group(2).strip().upper()}")
                        continue
                    m = re.match(r"^(\d+)\.\s+(.+)$", line)
                    if m and len(m.group(1)) <= 2:
                        # Don't treat as domain when we're inside a control and line looks like criteria (e.g. "2. Mandate the requirements...")
                        if pending_id and len((m.group(2) or "").strip()) > 15:
                            pass  # fall through to add as demand line
                        else:
                            num = m.group(1)
                            current_domain = DOMAIN_NAMES.get(int(num), f"{num}. {m.group(2).upper()}")
                            continue
                    # Control heading first (AC 1.1 ...), then sub-domain (AC 1 ...)
                    ch = CONTROL_HEADING_PATTERN.match(line)
                    if ch:
                        flush_control()
                        prefix, num_dot, rest = ch.group(1), ch.group(2), ch.group(3).strip()
                        pending_id = f"{prefix.upper()} {num_dot}"
                        # Applicability from [B]/[T]/[A] at end of rest
                        app_b = APPLICABILITY_BRACKET.search(rest)
                        if app_b:
                            pending_applicability = BRACKET_TO_APPLICABILITY.get(
                                app_b.group(1).upper(), "Basic"
                            )
                            rest = APPLICABILITY_BRACKET.sub("", rest).strip()
                        pending_title = rest
                        demand_lines = []
                        # Same-page criteria: PDF often has criteria column before control column
                        pending_criteria_from_page = _criteria_block_from_page_lines(page_lines[:idx])
                        continue
                    # Sub-domain: "AC 1 Access Control Policy" -> store descriptive part only
                    sm = SUB_DOMAIN_PATTERN.match(line)
                    if sm:
                        flush_control()
                        current_sub_domain = _sub_domain_label(line)
                        current_refs = []
                        continue
                    ref_m = UAE_REF_PATTERN.match(line)
                    if ref_m:
                        current_refs = _parse_references(ref_m.group(1))
                        continue
                    # Body paragraph: add to current control demand (include short list lines "1.", "a.", "b.")
                    if pending_id and not line.startswith("Page "):
                        if len(line) > 5 or re.match(r"^[1-9][\.\)]\s*$", line) or re.match(r"^[a-z][\.\)]\s*$", line, re.I):
                            demand_lines.append(line)
                            if len(demand_lines) > max_demand_lines:
                                demand_lines = demand_lines[:max_demand_lines]
        flush_control()
        return controls

    def _build_output(self, controls: List[Dict]) -> Dict[str, Any]:
        return {
            "standard_name": self.standard_name,
            "version": self.version,
            "controls": controls,
        }

    def save(self, output: Dict[str, Any], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        n = len(output.get("controls", []))
        print(f"✓ Saved {n} controls to {output_path}")

    def save_control_ids_txt(self, output: Dict[str, Any], txt_path: str):
        controls = output.get("controls", [])
        ids = sorted({c["control_id"] for c in controls if c.get("control_id")})
        Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("# ADHICS control IDs (from extract_adhic_controls.py)\n")
            f.write("\n".join(ids))
        print(f"✓ Wrote {len(ids)} control IDs to {txt_path}")


def _find_col(header: List[str], names: List[str]) -> Optional[int]:
    for n in names:
        for i, h in enumerate(header):
            if n in (h or ""):
                return i
    return None


def _domain_for_sub_domain(sub_domain: str) -> Optional[str]:
    """Infer domain from prefix so HR 4 stays under Human Resources."""
    m = SUB_DOMAIN_PATTERN.match(sub_domain.strip())
    if m:
        num = PREFIX_TO_DOMAIN.get(m.group(1).upper())
        if num is not None:
            return DOMAIN_NAMES.get(num, "")
    return None


def _sub_domain_to_control_id(sub_domain: str) -> str:
    """e.g. 'HR 1 Human Resources Security Policy' -> 'HR 1.1'"""
    m = SUB_DOMAIN_PATTERN.match(sub_domain.strip())
    if m:
        return f"{m.group(1).upper()} {m.group(2)}.1"
    return sub_domain[:20].replace(" ", "_")


def _split_criteria(text: str) -> List[str]:
    if not text:
        return []
    items = re.split(r"\s*(?:\d+[\.\)]\s*|[a-z][\.\)]\s*)", text)
    return [s.strip() for s in items if len(s.strip()) > 10]


# Numbered list: "1." or "1)" at start of line; lettered sub: "a." or "a)" at start
_TOP_LEVEL_NUM = re.compile(r"^\s*(\d+)[\.\)]\s*(.*)$", re.IGNORECASE)
_LETTERED_SUB = re.compile(r"^\s*([a-z])[\.\)]\s*(.*)$", re.IGNORECASE)
# Inline: split by " 1. " " 2. " or " a. " " b. " when criteria are in one long line
_INLINE_TOP = re.compile(r"\s+(?=\d+[\.\)]\s+)")
_INLINE_LETTERED = re.compile(r"\s+(?=[a-z][\.\)]\s+)", re.IGNORECASE)


def _parse_nested_criteria_inline(text: str) -> List[Dict[str, Any]]:
    """
    Parse when criteria are in one line: "1. Define... a. Background b. Roles 2. Mandate...".
    Splits by top-level numbers then by lettered sub-items.
    """
    if not text or not text.strip():
        return []
    # Strip leading "The policy shall:" or similar
    text = re.sub(r"^(?:The policy shall|The healthcare entity shall)\s*:?\s*", "", text, flags=re.IGNORECASE).strip()
    # Split by " 1. " " 2. " " 3. " (lookahead so we keep the delimiter content on the right)
    parts = _INLINE_TOP.split(text)
    out: List[Dict[str, Any]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Part is like "1. Define management requirements on; a. Background b. Roles" or "2. Mandate..."
        # Remove leading digit. or digit)
        part = re.sub(r"^\d+[\.\)]\s*", "", part).strip()
        if not part:
            continue
        # Split by " a. " " b. " etc. to get main text and sub-items
        sub_parts = _INLINE_LETTERED.split(part)
        main_text = (sub_parts[0] or "").strip()
        sub_items: List[str] = []
        for i in range(1, len(sub_parts)):
            # Remove leading "a. " etc.
            sub_text = re.sub(r"^[a-z][\.\)]\s*", "", (sub_parts[i] or "").strip(), flags=re.IGNORECASE).strip()
            if sub_text:
                sub_items.append(sub_text)
        out.append({"text": main_text, "sub": sub_items})
    return out


def _parse_nested_criteria(text: str) -> List[Dict[str, Any]]:
    """
    Parse criteria with nested structure. Top-level: 1. 2. 3. Sub-level: a. b. c.
    Supports both newline-separated lines and inline (single-line) format.
    Returns list of { "text": str, "sub": [str] }.
    """
    if not text or not text.strip():
        return []
    out: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        # Top-level: line starts with optional spaces then digit then . or )
        top = _TOP_LEVEL_NUM.match(raw_line)
        if top:
            if current and (current.get("text") or "").strip():
                out.append(current)
            current = {"text": top.group(2).strip(), "sub": []}
            continue
        # Lettered sub-item: a. b. c. (single letter at start of line)
        sub = _LETTERED_SUB.match(raw_line)
        if sub and current is not None:
            current["sub"] = current.get("sub") or []
            current["sub"].append(sub.group(2).strip())
            continue
        # Continuation of top-level or last sub-item
        if current is not None and not current.get("sub"):
            current["text"] = (current.get("text", "") + " " + stripped).strip()
        elif current is not None and current.get("sub"):
            last_sub = current["sub"][-1]
            current["sub"][-1] = (last_sub + " " + stripped).strip()
    if current and (current.get("text") or "").strip():
        out.append(current)
    # If no structure from newlines, or we have criteria but no sub-items and text contains " a. ", try inline
    has_lettered = bool(re.search(r"\s+[a-z][\.\)]\s+", text, re.IGNORECASE))
    if not out and re.search(r"\d+[\.\)]\s+", text) and len(text) > 20:
        out = _parse_nested_criteria_inline(text)
    elif out and has_lettered and not any((item.get("sub") or []) for item in out):
        # Line-by-line gave us items but no sub; reparse whole text as inline to get sub
        inline = _parse_nested_criteria_inline(text)
        if inline and any((item.get("sub") or []) for item in inline):
            out = inline
    return out


def _parse_description_and_criteria(demand_lines: List[str], pending_title: str) -> tuple:
    """
    Split body into: (1) control description (the main 'shall' statement), (2) nested criteria list.
    Preserves newlines so nested list (1. / a. / b.) can be parsed.
    """
    # Use pending_title as description when it is already the full shall-statement
    title = (pending_title or "").strip()
    if title and "shall" in title.lower() and len(title) > 30:
        description_from_title = title
    else:
        description_from_title = ""

    # If title doesn't end with period, first line(s) of demand may be continuation of the sentence
    lines = list(demand_lines) if demand_lines else []
    while description_from_title and not description_from_title.endswith(".") and lines:
        first = lines[0].strip()
        if not first:
            lines.pop(0)
            continue
        if re.match(r"^[12]\.\s+", first) or re.match(r"^The policy shall\s*:", first, re.I) or re.match(
            r"^The healthcare entity shall\s*:", first, re.I
        ):
            break
        description_from_title = (description_from_title + " " + first).strip()
        lines.pop(0)
        if description_from_title.endswith("."):
            break

    full = "\n".join(lines).strip() if lines else ""
    if not full and description_from_title:
        if not description_from_title.endswith("."):
            description_from_title += "."
        return description_from_title, []

    # Find where criteria start: numbered list "1." / "2." or "The policy shall:" / "The healthcare entity shall:"
    criteria_start = len(full)
    for pat in [
        r"^The policy shall\s*:",
        r"^The healthcare entity shall\s*:",
        r"\nThe policy shall\s*:",
        r"\nThe healthcare entity shall\s*:",
        r"\n\s*1\.\s+",
        r"\n\s*2\.\s+",
        r"\n\s*1\)\s+",
        r"\s+1\.\s+",
        r"\s+2\.\s+",
        r"\nThe healthcare entity,\s*",
    ]:
        m = re.search(pat, full, re.IGNORECASE)
        if m:
            criteria_start = min(criteria_start, m.start())

    if criteria_start < len(full):
        intro = full[:criteria_start].strip().replace("\n", " ") if criteria_start > 0 else ""
        if not description_from_title and intro:
            for sent in re.split(r"\.\s+(?=[A-Z])", intro):
                sent = sent.strip()
                if sent and "shall" in sent.lower() and len(sent) > 30:
                    description_from_title = sent if sent.endswith(".") else sent + "."
                    break
            if not description_from_title:
                description_from_title = intro[:500].strip()
                if not description_from_title.endswith("."):
                    description_from_title += "."
        criteria_text = full[criteria_start:].strip()
    else:
        intro = full.replace("\n", " ").strip()
        if not description_from_title and intro and "shall" in intro.lower():
            first_sent = re.split(r"\.\s+(?=[A-Z])", intro, maxsplit=1)
            if first_sent:
                description_from_title = first_sent[0].strip()
                if not description_from_title.endswith("."):
                    description_from_title += "."
        criteria_text = ""

    if description_from_title and not description_from_title.endswith("."):
        description_from_title += "."
    description = description_from_title or title or ""
    criteria = _parse_nested_criteria(criteria_text) if criteria_text else []
    return description, criteria


def main():
    import argparse
    p = argparse.ArgumentParser(description="Extract ADHICS controls from PDF (domain, sub_domain, control_id, control_demand, applicability, criteria, references)")
    p.add_argument("--pdf", default="data/01_raw/regulation/ADHIC.pdf", help="Path to ADHIC PDF")
    p.add_argument("--output", default="data/02_processed/adhics_controls_structured.json", help="Output JSON path")
    p.add_argument("--update-control-ids", action="store_true", help="Update adhics_control_ids.txt")
    args = p.parse_args()

    extractor = ADHICControlExtractor()
    output = extractor.extract_from_pdf(args.pdf)
    extractor.save(output, args.output)
    if args.update_control_ids:
        extractor.save_control_ids_txt(output, "data/03_label_studio_input/adhics_control_ids.txt")

    print("\nNext: run build_policy_controls_label_studio_xml.py to refresh Label Studio ADHICS dropdown.")


if __name__ == "__main__":
    main()
