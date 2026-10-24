import os
from pathlib import Path
from typing import Dict, List, Optional
from configs.settings import SAMPLE_CLAIMS_DIR

class EvidenceStore:
    """Manages dossier files, reading documents safely without executing them."""
    
    def __init__(self, claims_dir: Path = SAMPLE_CLAIMS_DIR):
        self.claims_dir = claims_dir

    def get_claim_folder(self, claim_id: str) -> Optional[Path]:
        # 1. Direct match on directory name
        for folder in self.claims_dir.iterdir():
            if folder.is_dir() and (folder.name == claim_id or folder.name.startswith(claim_id)):
                return folder
                
        # 2. Match via claim.json inside directory
        for folder in self.claims_dir.iterdir():
            if folder.is_dir():
                claim_json_path = folder / "claim.json"
                if claim_json_path.exists():
                    try:
                        import json
                        with open(claim_json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if data.get("claim_id") == claim_id:
                                return folder
                    except Exception:
                        pass
        return None

    def list_documents(self, claim_id: str) -> List[str]:
        folder = self.get_claim_folder(claim_id)
        if not folder:
            return []
        return [f.name for f in folder.iterdir() if f.is_file() and not f.name.endswith(".json")]

    def read_document(self, claim_id: str, filename: str) -> Optional[str]:
        folder = self.get_claim_folder(claim_id)
        if not folder:
            return None
        file_path = folder / filename
        if not file_path.exists() or not file_path.is_file():
            return None
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def get_all_claim_documents(self, claim_id: str) -> Dict[str, str]:
        docs = {}
        for fname in self.list_documents(claim_id):
            content = self.read_document(claim_id, fname)
            if content is not None:
                docs[fname] = content
        return docs
