"""
Dataset Ingestion & Metadata Normalizer for ai4bharat/MSMARCO-XI
"""
import json
import os
from typing import List, Dict, Any

class MSMARCOLoader:
    def __init__(self, data_path: str = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, 'data', 'msmarco_xi_sample.json')
        self.data_path = data_path
        self.documents: List[Dict[str, Any]] = []
        self.load_dataset()

    def load_dataset(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        else:
            self.documents = []
        return self.documents

    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        for doc in self.documents:
            if doc['doc_id'] == doc_id:
                return doc
        return {}

    def get_all_documents(self) -> List[Dict[str, Any]]:
        return self.documents
