"""
Configuration management for the AI Agent Matching System
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MatchingWeights:
    """Configuration for matching algorithm weights"""
    domain_alignment: float = 0.4
    technology_overlap: float = 0.3
    complexity_match: float = 0.2
    cost_efficiency: float = 0.1

@dataclass
class CostOptimizationRules:
    """Configuration for cost optimization"""
    prefer_free_tiers: bool = False
    max_monthly_ai_cost: int = 300
    prioritize_open_source: bool = False
    human_expert_hours_limit: int = 80

@dataclass
class SystemConfig:
    """Main system configuration"""
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Data paths
    data_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # Matching algorithm
    matching_weights: MatchingWeights = None
    
    # Cost optimization
    cost_rules: CostOptimizationRules = None
    
    # Performance
    max_agents_returned: int = 10
    max_experts_returned: int = 5
    cache_recommendations: bool = True
    cache_ttl_hours: int = 24
    
    # API settings
    api_timeout_seconds: int = 30
    rate_limit_per_minute: int = 60
    
    # Feature flags
    enable_ml_matching: bool = True
    enable_cost_optimization: bool = True
    enable_risk_assessment: bool = True
    enable_feedback_learning: bool = False
    
    def __post_init__(self):
        if self.matching_weights is None:
            self.matching_weights = MatchingWeights()
        if self.cost_rules is None:
            self.cost_rules = CostOptimizationRules()

class ConfigManager:
    """Manages system configuration with environment variable overrides"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from file and environment variables"""
        config = SystemConfig()
        
        # Load from environment variables
        config.log_level = os.getenv("AI_MATCHING_LOG_LEVEL", config.log_level)
        config.data_dir = os.getenv("AI_MATCHING_DATA_DIR", config.data_dir)
        config.cache_dir = os.getenv("AI_MATCHING_CACHE_DIR", config.cache_dir)
        
        # Performance settings
        config.max_agents_returned = int(os.getenv("AI_MATCHING_MAX_AGENTS", config.max_agents_returned))
        config.max_experts_returned = int(os.getenv("AI_MATCHING_MAX_EXPERTS", config.max_experts_returned))
        
        # Feature flags
        config.enable_ml_matching = os.getenv("AI_MATCHING_ENABLE_ML", "true").lower() == "true"
        config.enable_cost_optimization = os.getenv("AI_MATCHING_ENABLE_COST_OPT", "true").lower() == "true"
        config.enable_risk_assessment = os.getenv("AI_MATCHING_ENABLE_RISK", "true").lower() == "true"
        
        # API settings
        config.api_timeout_seconds = int(os.getenv("AI_MATCHING_API_TIMEOUT", config.api_timeout_seconds))
        config.rate_limit_per_minute = int(os.getenv("AI_MATCHING_RATE_LIMIT", config.rate_limit_per_minute))
        
        # Matching weights (from environment)
        if os.getenv("AI_MATCHING_DOMAIN_WEIGHT"):
            config.matching_weights.domain_alignment = float(os.getenv("AI_MATCHING_DOMAIN_WEIGHT"))
        if os.getenv("AI_MATCHING_TECH_WEIGHT"):
            config.matching_weights.technology_overlap = float(os.getenv("AI_MATCHING_TECH_WEIGHT"))
        if os.getenv("AI_MATCHING_COMPLEXITY_WEIGHT"):
            config.matching_weights.complexity_match = float(os.getenv("AI_MATCHING_COMPLEXITY_WEIGHT"))
        if os.getenv("AI_MATCHING_COST_WEIGHT"):
            config.matching_weights.cost_efficiency = float(os.getenv("AI_MATCHING_COST_WEIGHT"))
        
        # Cost optimization (from environment)
        if os.getenv("AI_MATCHING_PREFER_FREE"):
            config.cost_rules.prefer_free_tiers = os.getenv("AI_MATCHING_PREFER_FREE").lower() == "true"
        if os.getenv("AI_MATCHING_MAX_COST"):
            config.cost_rules.max_monthly_ai_cost = int(os.getenv("AI_MATCHING_MAX_COST"))
        
        return config
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        logging.basicConfig(
            level=getattr(logging, self._config.log_level.upper()),
            format=self._config.log_format
        )
    
    @property
    def config(self) -> SystemConfig:
        """Get current configuration"""
        return self._config
    
    def get_data_dir(self) -> Path:
        """Get data directory path"""
        if self._config.data_dir:
            return Path(self._config.data_dir)
        else:
            # Default to data directory relative to project root
            return Path(__file__).parent.parent / "data"
    
    def get_cache_dir(self) -> Path:
        """Get cache directory path"""
        if self._config.cache_dir:
            return Path(self._config.cache_dir)
        else:
            # Default to cache directory in project root
            cache_dir = Path(__file__).parent.parent / "cache"
            cache_dir.mkdir(exist_ok=True)
            return cache_dir
    
    def update_config(self, **kwargs):
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                logging.warning(f"Unknown configuration key: {key}")
    
    def validate_config(self) -> list:
        """Validate configuration and return any errors"""
        errors = []
        
        # Validate matching weights sum to 1.0
        total_weight = (
            self._config.matching_weights.domain_alignment +
            self._config.matching_weights.technology_overlap +
            self._config.matching_weights.complexity_match +
            self._config.matching_weights.cost_efficiency
        )
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Matching weights sum to {total_weight:.3f}, should be 1.0")
        
        # Validate cost limits
        if self._config.cost_rules.max_monthly_ai_cost < 0:
            errors.append("max_monthly_ai_cost cannot be negative")
        
        # Validate performance settings
        if self._config.max_agents_returned < 1:
            errors.append("max_agents_returned must be at least 1")
        if self._config.max_experts_returned < 1:
            errors.append("max_experts_returned must be at least 1")
        
        # Validate API settings
        if self._config.api_timeout_seconds < 1:
            errors.append("api_timeout_seconds must be at least 1")
        if self._config.rate_limit_per_minute < 1:
            errors.append("rate_limit_per_minute must be at least 1")
        
        return errors

# Global configuration instance
_config_manager = None

def get_config() -> SystemConfig:
    """Get global configuration instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def reset_config():
    """Reset global configuration (useful for testing)"""
    global _config_manager
    _config_manager = None

# Environment-specific configurations
class DevelopmentConfig(SystemConfig):
    """Development environment configuration"""
    def __init__(self):
        super().__init__()
        self.log_level = "DEBUG"
        self.cache_recommendations = False
        self.enable_feedback_learning = True

class ProductionConfig(SystemConfig):
    """Production environment configuration"""
    def __init__(self):
        super().__init__()
        self.log_level = "WARNING"
        self.cache_recommendations = True
        self.enable_feedback_learning = True
        self.rate_limit_per_minute = 120

class TestingConfig(SystemConfig):
    """Testing environment configuration"""
    def __init__(self):
        super().__init__()
        self.log_level = "ERROR"
        self.cache_recommendations = False
        self.max_agents_returned = 3
        self.max_experts_returned = 2

def get_environment_config(env: str = None) -> SystemConfig:
    """Get configuration for specific environment"""
    if env is None:
        env = os.getenv("AI_MATCHING_ENV", "development")
    
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig,
    }
    
    config_class = configs.get(env.lower(), SystemConfig)
    return config_class()

if __name__ == "__main__":
    # Test configuration loading
    config_manager = ConfigManager()
    config = config_manager.config
    
    print(f"Log level: {config.log_level}")
    print(f"Data directory: {config_manager.get_data_dir()}")
    print(f"Cache directory: {config_manager.get_cache_dir()}")
    print(f"Matching weights: {config.matching_weights}")
    print(f"Cost rules: {config.cost_rules}")
    
    # Validate configuration
    errors = config_manager.validate_config()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration validation passed")