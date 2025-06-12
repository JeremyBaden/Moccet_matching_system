"""
AI Agent Matching System

An intelligent matching system that analyzes project requirements and recommends
the optimal combination of AI agents and human experts for maximum project success.
"""

__version__ = "1.0.0"
__author__ = "AI Agent Matching System Team"
__email__ = "support@ai-agent-matching.com"

# Core classes and functions
from .matching_system import (
    ProjectRequirementAnalyzer,
    MatchResult,
    HumanExpert
)

from .data_loader import (
    DataLoader,
    load_agent_catalog,
    load_matching_config,
    load_expert_profiles
)

from .config import (
    get_config,
    get_config_manager,
    SystemConfig,
    MatchingWeights,
    CostOptimizationRules
)

# Main API
def analyze_project(project_brief: str, agent_catalog: list = None):
    """
    Quick analysis function for simple use cases.
    
    Args:
        project_brief: Description of the project requirements
        agent_catalog: Optional custom agent catalog (uses default if None)
    
    Returns:
        Dictionary containing analysis results and recommendations
    """
    if agent_catalog is None:
        agent_catalog = load_agent_catalog()
    
    analyzer = ProjectRequirementAnalyzer()
    return analyzer.generate_comprehensive_recommendation(project_brief, agent_catalog)

def quick_match(project_brief: str, top_n: int = 3):
    """
    Quick matching function that returns only top agent recommendations.
    
    Args:
        project_brief: Description of the project requirements
        top_n: Number of top agents to return (default: 3)
    
    Returns:
        List of top N agent recommendations
    """
    catalog = load_agent_catalog()
    analyzer = ProjectRequirementAnalyzer()
    analysis = analyzer.extract_keywords_and_analyze(project_brief)
    matches = analyzer.match_agents_to_requirements(analysis, catalog)
    
    return [{
        'name': match.agent_name,
        'confidence': round(match.confidence_score, 2),
        'priority': match.priority,
        'reasoning': match.reasoning
    } for match in matches[:top_n]]

# Package metadata
__all__ = [
    # Core classes
    'ProjectRequirementAnalyzer',
    'MatchResult', 
    'HumanExpert',
    'DataLoader',
    'SystemConfig',
    'MatchingWeights',
    'CostOptimizationRules',
    
    # Data loading functions
    'load_agent_catalog',
    'load_matching_config',
    'load_expert_profiles',
    
    # Configuration functions
    'get_config',
    'get_config_manager',
    
    # Convenience functions
    'analyze_project',
    'quick_match'
]

# Package information
def get_version():
    """Get package version."""
    return __version__

def get_info():
    """Get package information."""
    return {
        'name': 'ai-agent-matching-system',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': 'Intelligent matching system for AI agents and human experts'
    }
