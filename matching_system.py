import json
import re
from collections import defaultdict
from typing import List, Dict

class MatchingSystem:
    def __init__(self, catalog_path: str):
        with open(catalog_path, "r") as f:
            self.catalog = json.load(f)

    def extract_keywords(self, text: str) -> List[str]:
        stopwords = {"the", "and", "of", "to", "a", "in", "for", "on", "with", "is", "by", "that", "as", "an", "at"}
        words = re.findall(r"\b\w+\b", text.lower())
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return list(set(keywords))

    def match_requirements(self, project_description: str) -> Dict[str, List[Dict]]:
        keywords = self.extract_keywords(project_description)
        matches = defaultdict(list)

        for agent in self.catalog.get("agents", []):
            agent_keywords = set(agent.get("expertise", []))
            score = len(set(keywords) & agent_keywords)
            if score > 0:
                matches["ai_agents"].append({
                    "name": agent["name"],
                    "score": score,
                    "matched_keywords": list(set(keywords) & agent_keywords)
                })

        for expert in self.catalog.get("experts", []):
            expert_keywords = set(expert.get("skills", []))
            score = len(set(keywords) & expert_keywords)
            if score > 0:
                matches["human_experts"].append({
                    "name": expert["name"],
                    "score": score,
                    "matched_keywords": list(set(keywords) & expert_keywords)
                })

        matches["ai_agents"] = sorted(matches["ai_agents"], key=lambda x: x["score"], reverse=True)
        matches["human_experts"] = sorted(matches["human_experts"], key=lambda x: x["score"], reverse=True)
        return matches 
