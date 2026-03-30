# core/interfaces.py
from typing import Dict, List, Any


class AnalysisPipeline:
    """Abstract interface enforcing strict boundaries between API and Data processing."""

    def run(self, image_files: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Pipeline must implement run()")