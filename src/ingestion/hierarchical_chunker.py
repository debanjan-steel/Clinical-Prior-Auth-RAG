"""
Hierarchical Section Chunker & Clinical Policy Parser.
Extracts structured policy chunks while preserving parent section breadcrumbs,
procedure codes (CPT/HCPCS), and diagnosis codes (ICD-10).
"""

from dataclasses import dataclass, field, asdict
import re
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class PolicyChunk:
    chunk_id: str
    policy_id: str
    policy_title: str
    section_hierarchy: str
    content: str
    full_text_for_embedding: str
    cpt_codes: List[str] = field(default_factory=list)
    icd_codes: List[str] = field(default_factory=list)
    page_number: int = 1
    source_file: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HierarchicalPolicyChunker:
    """
    Parses Clinical Policy Bulletins (Markdown or Text) and generates hierarchical chunks
    with structural breadcrumbs to prevent loss of parent-child criteria context.
    """

    def __init__(self, max_chunk_tokens: int = 400, chunk_overlap_tokens: int = 50):
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def _extract_codes(self, text: str) -> tuple[List[str], List[str]]:
        """Extracts CPT/HCPCS and ICD-10 codes from policy text."""
        # CPT & HCPCS (e.g., CPT 22612, HCPCS J0135, CPT 96372)
        cpt_matches = re.findall(r'\b(?:CPT|HCPCS)\s*[:\s]?\s*([A-Z0-9,\s]+?)(?=\n|\.\s|;|$)', text, re.IGNORECASE)
        cpt_codes = set()
        for match in cpt_matches:
            for code in re.findall(r'\b[A-Z]?[0-9]{4,5}[A-Z]?\b', match):
                cpt_codes.add(code.strip())

        # ICD-10 codes (e.g. M54.5, M43.16, C34.90)
        icd_matches = re.findall(r'\b(?:ICD-10)\s*[:\s]?\s*([A-Z0-9,\.\s]+?)(?=\n|\.\s|;|$)', text, re.IGNORECASE)
        icd_codes = set()
        for match in icd_matches:
            for code in re.findall(r'\b[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?\b', match):
                icd_codes.add(code.strip())

        return sorted(list(cpt_codes)), sorted(list(icd_codes))

    def chunk_document(self, text: str, source_path: str = "") -> List[PolicyChunk]:
        """
        Parses policy document into hierarchical chunks by tracking markdown headers
        and associating contextual breadcrumbs with each child segment.
        """
        lines = text.splitlines()
        chunks: List[PolicyChunk] = []

        policy_id = "UNKNOWN-POLICY"
        policy_title = "Clinical Policy"
        global_cpt: List[str] = []
        global_icd: List[str] = []

        # Extract top-level metadata
        for line in lines[:20]:
            if line.startswith("# "):
                policy_title = line.replace("# ", "").strip()
            elif "Policy Number:" in line or "Policy ID:" in line:
                policy_id = line.split(":", 1)[1].strip()

        # Extract any globally declared codes in headers
        top_header_text = "\n".join(lines[:25])
        extracted_cpt, extracted_icd = self._extract_codes(top_header_text)
        global_cpt.extend(extracted_cpt)
        global_icd.extend(extracted_icd)

        # Structure tracker for breadcrumb hierarchy
        current_h1 = policy_title
        current_h2 = ""
        current_h3 = ""
        current_buffer: List[str] = []
        current_page = 1

        def flush_buffer():
            if not current_buffer:
                return

            content_text = "\n".join(current_buffer).strip()
            if not content_text:
                return

            # Construct breadcrumb string
            breadcrumb_parts = [f"Policy: {policy_id} - {policy_title}"]
            if current_h2:
                breadcrumb_parts.append(current_h2)
            if current_h3:
                breadcrumb_parts.append(current_h3)
            hierarchy_str = " > ".join(breadcrumb_parts)

            # Local code extraction + inherited global codes
            local_cpt, local_icd = self._extract_codes(content_text)
            combined_cpt = sorted(list(set(global_cpt + local_cpt)))
            combined_icd = sorted(list(set(global_icd + local_icd)))

            # Format full text with embedded metadata for robust semantic search
            codes_context = ""
            if combined_cpt:
                codes_context += f" [Applicable CPT: {', '.join(combined_cpt)}]"
            if combined_icd:
                codes_context += f" [Applicable ICD-10: {', '.join(combined_icd)}]"

            full_text = f"[{hierarchy_str}]{codes_context}\n{content_text}"

            chunk_id = f"{policy_id}_{len(chunks) + 1:03d}"
            chunks.append(
                PolicyChunk(
                    chunk_id=chunk_id,
                    policy_id=policy_id,
                    policy_title=policy_title,
                    section_hierarchy=hierarchy_str,
                    content=content_text,
                    full_text_for_embedding=full_text,
                    cpt_codes=combined_cpt,
                    icd_codes=combined_icd,
                    page_number=current_page,
                    source_file=Path(source_path).name if source_path else "memory",
                )
            )
            current_buffer.clear()

        for line in lines:
            # Check for page markers
            if "--- Page" in line or "[Page " in line:
                page_match = re.search(r'Page\s+(\d+)', line)
                if page_match:
                    current_page = int(page_match.group(1))

            if line.startswith("# "):
                flush_buffer()
                current_h1 = line.replace("# ", "").strip()
                current_h2 = ""
                current_h3 = ""
            elif line.startswith("## "):
                flush_buffer()
                current_h2 = line.replace("## ", "").strip()
                current_h3 = ""
            elif line.startswith("### "):
                flush_buffer()
                current_h3 = line.replace("### ", "").strip()
            else:
                current_buffer.append(line)

        flush_buffer()
        return chunks

    def chunk_file(self, file_path: str | Path) -> List[PolicyChunk]:
        """Reads file (PDF, TXT, MD) and generates hierarchical policy chunks."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages_text = []
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                pages_text.append(f"\n--- Page {idx + 1} ---\n{page_text}")
            full_text = "\n".join(pages_text)
        else:
            full_text = path.read_text(encoding="utf-8")

        return self.chunk_document(full_text, source_path=str(path))
