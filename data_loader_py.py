"""
Data loading and management utilities for the AI Agent Matching System
"""

import json
import csv
import os
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and management of agent catalogs and configuration data"""
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            # Default to data directory relative to this file
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.agents_file = self.data_dir / "agents_catalog.json"
        self.config_file = self.data_dir / "matching_config.json"
        self.experts_file = self.data_dir / "expert_profiles.json"
    
    def load_agent_catalog(self) -> List[Dict]:
        """Load the complete AI agent catalog"""
        try:
            with open(self.agents_file, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
            logger.info(f"Loaded {len(catalog)} agents from catalog")
            return catalog
        except FileNotFoundError:
            logger.error(f"Agent catalog not found at {self.agents_file}")
            return self._get_default_catalog()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing agent catalog: {e}")
            return self._get_default_catalog()
    
    def load_matching_config(self) -> Dict:
        """Load matching system configuration"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("Loaded matching configuration")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_file}, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}")
            return self._get_default_config()
    
    def load_expert_profiles(self) -> Dict:
        """Load human expert profiles"""
        try:
            with open(self.experts_file, 'r', encoding='utf-8') as f:
                experts = json.load(f)
            logger.info(f"Loaded {len(experts)} expert profiles")
            return experts
        except FileNotFoundError:
            logger.warning(f"Expert profiles not found at {self.experts_file}, using defaults")
            return self._get_default_experts()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing expert profiles: {e}")
            return self._get_default_experts()
    
    def save_agent_catalog(self, catalog: List[Dict]) -> bool:
        """Save updated agent catalog"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(self.agents_file, 'w', encoding='utf-8') as f:
                json.dump(catalog, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(catalog)} agents to catalog")
            return True
        except Exception as e:
            logger.error(f"Error saving agent catalog: {e}")
            return False
    
    def save_matching_config(self, config: Dict) -> bool:
        """Save matching configuration"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info("Saved matching configuration")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def validate_catalog(self, catalog: List[Dict]) -> List[str]:
        """Validate agent catalog structure and return any errors"""
        errors = []
        required_fields = ['name', 'provider', 'type', 'capabilities', 'limitations', 'integration', 'ideal_use_cases']
        
        for i, agent in enumerate(catalog):
            if not isinstance(agent, dict):
                errors.append(f"Agent {i}: Must be a dictionary")
                continue
            
            for field in required_fields:
                if field not in agent:
                    errors.append(f"Agent {i} ({agent.get('name', 'Unknown')}): Missing required field '{field}'")
                elif field in ['capabilities', 'limitations', 'integration', 'ideal_use_cases']:
                    if not isinstance(agent[field], list):
                        errors.append(f"Agent {i} ({agent.get('name', 'Unknown')}): Field '{field}' must be a list")
        
        return errors
    
    def _get_default_catalog(self) -> List[Dict]:
        """Return a minimal default catalog if main catalog can't be loaded"""
        return [
            {
                "name": "OpenAI GPT-4",
                "provider": "OpenAI",
                "type": "General-purpose LLM",
                "capabilities": ["complex code generation", "documentation", "debugging"],
                "limitations": ["higher cost", "rate-limited"],
                "integration": ["OpenAI API", "VS Code plugins"],
                "ideal_use_cases": ["complex backend logic", "documentation"],
                "pricing": {"prompt_tokens_per_1K": 0.03, "completion_tokens_per_1K": 0.06}
            },
            {
                "name": "GitHub Copilot",
                "provider": "GitHub (Microsoft/OpenAI)",
                "type": "AI pair-programmer",
                "capabilities": ["inline code suggestions", "boilerplate generation"],
                "limitations": ["limited context awareness", "no standalone API"],
                "integration": ["VS Code", "JetBrains IDEs"],
                "ideal_use_cases": ["day-to-day coding", "frontend development"],
                "pricing": {"individual_per_month": 10, "business_per_month": 19}
            }
        ]
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            "matching_weights": {
                "domain_alignment": 0.4,
                "technology_overlap": 0.3,
                "complexity_match": 0.2,
                "cost_efficiency": 0.1
            },
            "cost_optimization_rules": {
                "budget_conscious": {
                    "prefer_free_tiers": True,
                    "max_monthly_ai_cost": 100
                },
                "balanced": {
                    "prefer_free_tiers": False,
                    "max_monthly_ai_cost": 300
                }
            }
        }
    
    def _get_default_experts(self) -> Dict:
        """Return default expert profiles"""
        return {
            "senior_fullstack_developer": {
                "role": "Senior Full-Stack Developer",
                "skills": ["architecture", "system design", "code review"],
                "experience_domains": ["frontend", "backend", "database"],
                "collaboration_agents": ["GitHub Copilot", "OpenAI GPT-4"],
                "hourly_rate_range": "$80-150/hour"
            },
            "ui_ux_designer": {
                "role": "UI/UX Designer", 
                "skills": ["user research", "wireframing", "prototyping"],
                "experience_domains": ["design", "frontend"],
                "collaboration_agents": ["Galileo AI", "Uizard Autodesigner"],
                "hourly_rate_range": "$60-120/hour"
            }
        }

def load_agent_catalog() -> List[Dict]:
    """Convenience function to load agent catalog"""
    loader = DataLoader()
    return loader.load_agent_catalog()

def load_matching_config() -> Dict:
    """Convenience function to load matching config"""
    loader = DataLoader()
    return loader.load_matching_config()

def load_expert_profiles() -> Dict:
    """Convenience function to load expert profiles"""
    loader = DataLoader()
    return loader.load_expert_profiles()

# For backward compatibility
def load_full_agent_catalog() -> List[Dict]:
    """Alias for load_agent_catalog()"""
    return load_agent_catalog()

if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    
    catalog = loader.load_agent_catalog()
    print(f"Loaded {len(catalog)} agents")
    
    errors = loader.validate_catalog(catalog)
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Catalog validation passed")
    
    config = loader.load_matching_config()
    print(f"Loaded config with {len(config)} sections")
    
    experts = loader.load_expert_profiles()
    print(f"Loaded {len(experts)} expert profiles")