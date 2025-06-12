"""
AI Agent & Expert Matching System - Core Matching Algorithm

This module contains the core intelligence for analyzing project requirements
and matching them to optimal AI agents and human experts.
"""

import re
import json
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Result of matching an agent to project requirements"""
    agent_name: str
    confidence_score: float
    priority: int  # 1=Critical, 2=Important, 3=Helpful
    reasoning: str
    matched_keywords: List[str]
    estimated_monthly_cost: float = 0.0

@dataclass
class HumanExpert:
    """Human expert profile"""
    role: str
    skills: List[str]
    experience_domains: List[str]
    collaboration_agents: List[str]
    hourly_rate_range: str
    
class ProjectRequirementAnalyzer:
    """
    Intelligent analyzer that extracts requirements and matches them to
    optimal combinations of AI agents and human experts.
    """
    
    def __init__(self):
        self.domain_keywords = self._initialize_domain_keywords()
        self.complexity_indicators = self._initialize_complexity_indicators()
        self.urgency_keywords = self._initialize_urgency_keywords()
        self.technology_patterns = self._initialize_technology_patterns()
        self.human_experts = self._initialize_human_experts()
        
    def _initialize_domain_keywords(self) -> Dict[str, Set[str]]:
        """Initialize domain-specific keyword sets for classification"""
        return {
            'frontend': {
                'react', 'vue', 'angular', 'javascript', 'typescript', 'html', 'css', 
                'jsx', 'ui', 'ux', 'interface', 'responsive', 'mobile-first', 'design',
                'component', 'dom', 'browser', 'client-side', 'spa', 'pwa', 'webpack',
                'vite', 'nextjs', 'nuxt', 'svelte', 'bootstrap', 'tailwind', 'sass',
                'less', 'styled-components', 'material-ui', 'chakra', 'antd'
            },
            'backend': {
                'python', 'java', 'nodejs', 'django', 'flask', 'fastapi', 'spring',
                'express', 'api', 'rest', 'graphql', 'microservices', 'database',
                'sql', 'nosql', 'postgresql', 'mysql', 'mongodb', 'redis', 'server',
                'authentication', 'authorization', 'jwt', 'oauth', 'crud', 'orm',
                'middleware', 'routing', 'endpoint', 'controller', 'service', 'model',
                'migration', 'schema', 'query', 'transaction', 'caching'
            },
            'cloud': {
                'aws', 'azure', 'gcp', 'google cloud', 'cloud', 'serverless', 'lambda',
                'kubernetes', 'docker', 'container', 'deployment', 'devops', 'ci/cd',
                'pipeline', 'infrastructure', 'terraform', 'ansible', 'jenkins',
                'github actions', 'gitlab ci', 'monitoring', 'logging', 'metrics',
                'scaling', 'load balancer', 'cdn', 's3', 'ec2', 'rds', 'dynamodb',
                'cloudformation', 'elastic beanstalk', 'ecs', 'eks', 'fargate'
            },
            'mobile': {
                'ios', 'android', 'react native', 'flutter', 'swift', 'kotlin',
                'java', 'objective-c', 'mobile', 'app store', 'play store',
                'xamarin', 'ionic', 'cordova', 'phonegap', 'native', 'hybrid',
                'cross-platform', 'responsive', 'pwa', 'mobile-first'
            },
            'data': {
                'machine learning', 'ml', 'ai', 'data science', 'analytics', 'big data',
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
                'jupyter', 'python', 'r', 'statistics', 'visualization', 'dashboard',
                'etl', 'pipeline', 'warehouse', 'lake', 'spark', 'hadoop', 'kafka',
                'streamlit', 'plotly', 'matplotlib', 'seaborn', 'sql', 'nosql',
                'embedding', 'vector', 'llm', 'nlp', 'computer vision', 'deep learning'
            },
            'testing': {
                'testing', 'test', 'unit test', 'integration test', 'e2e', 'automation',
                'jest', 'pytest', 'junit', 'selenium', 'cypress', 'playwright',
                'test driven', 'tdd', 'bdd', 'coverage', 'mock', 'stub', 'fixture',
                'quality assurance', 'qa', 'bug', 'debugging', 'validation'
            },
            'security': {
                'security', 'authentication', 'authorization', 'encryption', 'https',
                'ssl', 'tls', 'oauth', 'jwt', 'csrf', 'xss', 'sql injection',
                'vulnerability', 'penetration testing', 'compliance', 'gdpr',
                'hipaa', 'sox', 'pci', 'audit', 'firewall', 'vpn', 'zero trust'
            },
            'design': {
                'design', 'ui', 'ux', 'user experience', 'user interface', 'wireframe',
                'mockup', 'prototype', 'figma', 'sketch', 'adobe xd', 'invision',
                'user research', 'persona', 'journey map', 'usability', 'accessibility',
                'a11y', 'design system', 'style guide', 'branding', 'logo', 'color',
                'typography', 'layout', 'grid', 'spacing', 'visual hierarchy'
            }
        }
    
    def _initialize_complexity_indicators(self) -> Dict[str, List[str]]:
        """Initialize complexity level indicators"""
        return {
            'high': [
                'enterprise', 'large-scale', 'distributed', 'microservices', 'complex',
                'sophisticated', 'advanced', 'mission-critical', 'high-availability',
                'fault-tolerant', 'scalable', 'multi-tenant', 'real-time', 'concurrent',
                'parallel processing', 'big data', 'machine learning', 'ai', 'blockchain',
                'compliance', 'security', 'encryption', 'audit', 'integration',
                'legacy system', 'migration', 'transformation', 'architecture'
            ],
            'medium': [
                'moderate', 'standard', 'typical', 'conventional', 'business logic',
                'workflow', 'process', 'management', 'dashboard', 'reporting',
                'analytics', 'integration', 'api', 'database', 'user management',
                'authentication', 'authorization', 'notification', 'email',
                'payment', 'e-commerce', 'crm', 'cms'
            ],
            'low': [
                'simple', 'basic', 'minimal', 'prototype', 'mvp', 'proof of concept',
                'demo', 'tutorial', 'learning', 'practice', 'small', 'lightweight',
                'static', 'single page', 'landing page', 'blog', 'portfolio',
                'todo', 'calculator', 'crud', 'form', 'list'
            ]
        }
    
    def _initialize_urgency_keywords(self) -> Dict[str, List[str]]:
        """Initialize urgency level keywords"""
        return {
            'urgent': [
                'urgent', 'asap', 'immediately', 'rush', 'critical', 'emergency',
                'deadline', 'quickly', 'fast', 'rapid', 'speed', 'time-sensitive',
                'launch date', 'go-live', 'yesterday', 'tight timeline'
            ],
            'normal': [
                'standard', 'normal', 'regular', 'typical', 'reasonable',
                'planned', 'scheduled', 'timeline', 'delivery', 'completion'
            ],
            'flexible': [
                'flexible', 'when possible', 'no rush', 'relaxed', 'whenever',
                'eventually', 'future', 'long-term', 'roadmap', 'backlog'
            ]
        }
    
    def _initialize_technology_patterns(self) -> List[Tuple[str, str]]:
        """Initialize technology detection patterns"""
        return [
            (r'\breact\b', 'react'),
            (r'\bvue\.?js?\b', 'vue'),
            (r'\bangular\b', 'angular'),
            (r'\bnode\.?js\b', 'nodejs'),
            (r'\bpython\b', 'python'),
            (r'\bdjango\b', 'django'),
            (r'\bflask\b', 'flask'),
            (r'\bfastapi\b', 'fastapi'),
            (r'\bjava\b(?!script)', 'java'),
            (r'\bspring\b', 'spring'),
            (r'\baws\b', 'aws'),
            (r'\bazure\b', 'azure'),
            (r'\bgcp\b|google cloud', 'gcp'),
            (r'\bdocker\b', 'docker'),
            (r'\bkubernetes\b|k8s', 'kubernetes'),
            (r'\bpostgresql\b|postgres', 'postgresql'),
            (r'\bmysql\b', 'mysql'),
            (r'\bmongodb\b|mongo', 'mongodb'),
            (r'\bredis\b', 'redis'),
            (r'\btypescript\b', 'typescript'),
            (r'\bjavascript\b', 'javascript'),
            (r'\bhtml\b', 'html'),
            (r'\bcss\b', 'css'),
            (r'\btailwind\b', 'tailwind'),
            (r'\bbootstrap\b', 'bootstrap'),
            (r'\bmachine learning\b|ml\b', 'machine_learning'),
            (r'\bai\b', 'ai'),
            (r'\btensorflow\b', 'tensorflow'),
            (r'\bpytorch\b', 'pytorch'),
            (r'\bfigma\b', 'figma'),
            (r'\bsketch\b', 'sketch')
        ]
    
    def _initialize_human_experts(self) -> List[HumanExpert]:
        """Initialize human expert profiles"""
        return [
            HumanExpert(
                role="Senior Full-Stack Developer",
                skills=["architecture", "system design", "code review", "mentoring", "technical leadership"],
                experience_domains=["frontend", "backend", "database", "cloud"],
                collaboration_agents=["GitHub Copilot", "OpenAI GPT-4", "Amazon CodeWhisperer"],
                hourly_rate_range="$80-150/hour"
            ),
            HumanExpert(
                role="UI/UX Designer",
                skills=["user research", "wireframing", "prototyping", "design systems", "user testing"],
                experience_domains=["design", "frontend", "mobile"],
                collaboration_agents=["Galileo AI", "Uizard Autodesigner", "OpenAI GPT-4"],
                hourly_rate_range="$60-120/hour"
            ),
            HumanExpert(
                role="DevOps Engineer",
                skills=["infrastructure", "ci/cd", "monitoring", "security", "automation"],
                experience_domains=["cloud", "backend", "security"],
                collaboration_agents=["Amazon CodeWhisperer", "OpenAI GPT-4", "GitHub Copilot"],
                hourly_rate_range="$90-160/hour"
            ),
            HumanExpert(
                role="QA Engineer",
                skills=["test planning", "automation", "performance testing", "quality assurance"],
                experience_domains=["testing", "backend", "frontend"],
                collaboration_agents=["Diffblue Cover", "Tabnine", "OpenAI GPT-4"],
                hourly_rate_range="$70-130/hour"
            ),
            HumanExpert(
                role="Data Scientist",
                skills=["machine learning", "data analysis", "modeling", "visualization"],
                experience_domains=["data", "backend", "cloud"],
                collaboration_agents=["OpenAI GPT-4", "Claude 2", "Hugging Face Transformers"],
                hourly_rate_range="$100-180/hour"
            ),
            HumanExpert(
                role="Technical Project Manager",
                skills=["project planning", "risk management", "team coordination", "stakeholder communication"],
                experience_domains=["management", "planning", "coordination"],
                collaboration_agents=["OpenAI GPT-4", "Claude 2", "Mintlify Docs AI"],
                hourly_rate_range="$80-140/hour"
            )
        ]
    
    def extract_keywords_and_analyze(self, project_brief: str) -> Dict[str, Any]:
        """
        Extract keywords and analyze project requirements to determine:
        - Domain scores (frontend, backend, cloud, etc.)
        - Complexity level
        - Urgency level
        - Technology stack
        - Team size estimates
        """
        # Normalize text
        text = project_brief.lower().strip()
        
        # Extract domain scores
        domain_scores = self._calculate_domain_scores(text)
        
        # Determine complexity
        complexity = self._determine_complexity(text, domain_scores)
        
        # Assess urgency
        urgency = self._assess_urgency(text)
        
        # Extract technologies
        technologies = self._extract_technologies(text)
        
        # Estimate team size
        team_size = self._estimate_team_size(domain_scores, complexity)
        
        # Extract budget and timeline if mentioned
        budget_info = self._extract_budget_info(text)
        timeline_info = self._extract_timeline_info(text)
        
        return {
            'domain_scores': domain_scores,
            'complexity': complexity,
            'urgency': urgency,
            'technologies': technologies,
            'team_size': team_size,
            'budget_info': budget_info,
            'timeline_info': timeline_info,
            'word_count': len(text.split()),
            'primary_domain': max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else 'backend'
        }
    
    def _calculate_domain_scores(self, text: str) -> Dict[str, float]:
        """Calculate relevance scores for each domain"""
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text)
            
            # Calculate weighted score
            if matches > 0:
                # Base score from match count
                base_score = min(matches / 10.0, 1.0)  # Cap at 1.0
                
                # Boost score based on keyword specificity
                specific_matches = sum(1 for keyword in keywords 
                                     if len(keyword) > 5 and keyword in text)
                specificity_boost = specific_matches * 0.1
                
                domain_scores[domain] = min(base_score + specificity_boost, 1.0)
            else:
                domain_scores[domain] = 0.0
        
        return domain_scores
    
    def _determine_complexity(self, text: str, domain_scores: Dict[str, float]) -> str:
        """Determine project complexity level"""
        # Count complexity indicators
        high_indicators = sum(1 for indicator in self.complexity_indicators['high'] 
                            if indicator in text)
        medium_indicators = sum(1 for indicator in self.complexity_indicators['medium'] 
                              if indicator in text)
        low_indicators = sum(1 for indicator in self.complexity_indicators['low'] 
                           if indicator in text)
        
        # Factor in domain diversity (more domains = higher complexity)
        active_domains = sum(1 for score in domain_scores.values() if score > 0.3)
        domain_complexity_factor = active_domains / len(domain_scores)
        
        # Calculate complexity score
        complexity_score = (
            high_indicators * 3 + 
            medium_indicators * 2 + 
            low_indicators * 1 +
            domain_complexity_factor * 5
        ) / max(1, len(text.split()) / 10)  # Normalize by text length
        
        # Determine complexity level
        if complexity_score >= 0.7 or high_indicators >= 2:
            return 'high'
        elif complexity_score >= 0.3 or medium_indicators >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _assess_urgency(self, text: str) -> str:
        """Assess project urgency level"""
        urgent_count = sum(1 for keyword in self.urgency_keywords['urgent'] 
                          if keyword in text)
        normal_count = sum(1 for keyword in self.urgency_keywords['normal'] 
                          if keyword in text)
        flexible_count = sum(1 for keyword in self.urgency_keywords['flexible'] 
                            if keyword in text)
        
        if urgent_count > 0:
            return 'urgent'
        elif flexible_count > normal_count:
            return 'flexible'
        else:
            return 'normal'
    
    def _extract_technologies(self, text: str) -> List[str]:
        """Extract mentioned technologies using pattern matching"""
        technologies = []
        
        for pattern, tech_name in self.technology_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                technologies.append(tech_name)
        
        return list(set(technologies))  # Remove duplicates
    
    def _estimate_team_size(self, domain_scores: Dict[str, float], complexity: str) -> Dict[str, int]:
        """Estimate required team size based on domain scores and complexity"""
        base_multiplier = {'low': 1, 'medium': 1.5, 'high': 2.5}[complexity]
        
        team_size = {
            'developers': max(1, int(sum(domain_scores.values()) * base_multiplier)),
            'designers': 1 if domain_scores.get('design', 0) > 0.3 or domain_scores.get('frontend', 0) > 0.5 else 0,
            'devops': 1 if domain_scores.get('cloud', 0) > 0.4 or complexity == 'high' else 0,
            'qa': 1 if complexity in ['medium', 'high'] else 0,
            'data_specialists': 1 if domain_scores.get('data', 0) > 0.5 else 0,
            'project_managers': 1 if complexity == 'high' or sum(domain_scores.values()) > 3 else 0
        }
        
        return team_size
    
    def _extract_budget_info(self, text: str) -> Dict[str, Any]:
        """Extract budget information from text"""
        budget_patterns = [
            r'\$([0-9,]+(?:\.[0-9]{2})?)[km]?',  # $10k, $50,000
            r'budget.*?([0-9,]+)',  # budget of 50000
            r'([0-9,]+).*?budget',  # 50000 budget
        ]
        
        budget_info = {'amount': None, 'currency': 'USD', 'type': 'total'}
        
        for pattern in budget_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount = float(amount_str)
                    # Handle k/m suffixes
                    if 'k' in text[match.start():match.end()]:
                        amount *= 1000
                    elif 'm' in text[match.start():match.end()]:
                        amount *= 1000000
                    budget_info['amount'] = amount
                    break
                except ValueError:
                    continue
        
        return budget_info
    
    def _extract_timeline_info(self, text: str) -> Dict[str, Any]:
        """Extract timeline information from text"""
        timeline_patterns = [
            r'(\d+)\s*weeks?',
            r'(\d+)\s*months?', 
            r'(\d+)\s*days?',
            r'(\d+)\s*years?'
        ]
        
        timeline_info = {'duration': None, 'unit': None, 'deadline': None}
        
        for pattern in timeline_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                duration = int(match.group(1))
                unit = 'weeks' if 'week' in match.group(0) else \
                       'months' if 'month' in match.group(0) else \
                       'days' if 'day' in match.group(0) else 'years'
                
                timeline_info['duration'] = duration
                timeline_info['unit'] = unit
                break
        
        return timeline_info
    
    def match_agents_to_requirements(self, analysis: Dict[str, Any], agent_catalog: List[Dict]) -> List[MatchResult]:
        """
        Match AI agents to project requirements with confidence scoring
        """
        matches = []
        
        for agent in agent_catalog:
            confidence_score, reasoning, matched_keywords = self._calculate_agent_match_score(
                agent, analysis
            )
            
            if confidence_score > 0.1:  # Only include agents with meaningful matches
                # Determine priority level
                priority = self._determine_agent_priority(confidence_score, agent, analysis)
                
                # Estimate monthly cost
                monthly_cost = self._estimate_agent_monthly_cost(agent, analysis)
                
                match = MatchResult(
                    agent_name=agent['name'],
                    confidence_score=confidence_score,
                    priority=priority,
                    reasoning=reasoning,
                    matched_keywords=matched_keywords,
                    estimated_monthly_cost=monthly_cost
                )
                matches.append(match)
        
        # Sort by priority (1 first) then by confidence score
        matches.sort(key=lambda x: (x.priority, -x.confidence_score))
        
        return matches
    
    def _calculate_agent_match_score(self, agent: Dict, analysis: Dict) -> Tuple[float, str, List[str]]:
        """Calculate how well an agent matches the project requirements"""
        
        # Extract agent keywords from various fields
        agent_keywords = set()
        for field in ['capabilities', 'ideal_use_cases', 'integration']:
            if field in agent:
                for item in agent[field]:
                    agent_keywords.update(item.lower().split())
        
        # Extract project keywords
        project_text = ' '.join(analysis['technologies']).lower()
        project_keywords = set(project_text.split())
        
        # Calculate domain alignment score
        domain_alignment_score = 0.0
        relevant_domains = []
        
        for domain, score in analysis['domain_scores'].items():
            if score > 0.2:  # Only consider domains with meaningful presence
                domain_keywords = self.domain_keywords[domain]
                agent_domain_overlap = len(agent_keywords.intersection(domain_keywords))
                if agent_domain_overlap > 0:
                    domain_contribution = score * (agent_domain_overlap / len(domain_keywords))
                    domain_alignment_score += domain_contribution
                    relevant_domains.append(domain)
        
        # Calculate technology overlap score
        tech_overlap_score = len(agent_keywords.intersection(project_keywords)) / max(len(project_keywords), 1)
        
        # Calculate complexity match score
        complexity_match_score = self._calculate_complexity_match(agent, analysis['complexity'])
        
        # Calculate use case alignment
        use_case_score = self._calculate_use_case_alignment(agent, analysis)
        
        # Weighted final score
        weights = {
            'domain_alignment': 0.4,
            'tech_overlap': 0.3,
            'complexity_match': 0.2,
            'use_case': 0.1
        }
        
        final_score = (
            domain_alignment_score * weights['domain_alignment'] +
            tech_overlap_score * weights['tech_overlap'] +
            complexity_match_score * weights['complexity_match'] +
            use_case_score * weights['use_case']
        )
        
        # Cap the score at 1.0
        final_score = min(final_score, 1.0)
        
        # Generate reasoning
        reasoning = self._generate_agent_reasoning(agent, analysis, relevant_domains, final_score)
        
        # Find matched keywords
        matched_keywords = list(agent_keywords.intersection(project_keywords))
        
        return final_score, reasoning, matched_keywords
    
    def _calculate_complexity_match(self, agent: Dict, complexity: str) -> float:
        """Calculate how well agent matches project complexity"""
        agent_name = agent['name'].lower()
        agent_type = agent.get('type', '').lower()
        
        # Map agent types to complexity suitability
        complexity_mapping = {
            'high': {
                'openai gpt-4': 0.9,
                'claude': 0.9,
                'amazon codewhisperer': 0.8,
                'github copilot': 0.7,
                'diffblue cover': 0.8,
                'enterprise': 0.9
            },
            'medium': {
                'github copilot': 0.9,
                'tabnine': 0.8,
                'replit ghostwriter': 0.7,
                'jetbrains ai': 0.8,
                'codeium': 0.8
            },
            'low': {
                'codeium': 0.9,
                'replit ghostwriter': 0.9,
                'tabnine': 0.7,
                'github copilot': 0.8
            }
        }
        
        # Check for direct matches
        for name_part, score in complexity_mapping.get(complexity, {}).items():
            if name_part in agent_name:
                return score
        
        # Default scoring based on type
        if complexity == 'high' and 'enterprise' in agent_type:
            return 0.8
        elif complexity == 'low' and 'free' in agent.get('pricing', {}):
            return 0.8
        else:
            return 0.5  # Neutral match
    
    def _calculate_use_case_alignment(self, agent: Dict, analysis: Dict) -> float:
        """Calculate alignment between agent use cases and project needs"""
        use_cases = agent.get('ideal_use_cases', [])
        
        alignment_score = 0.0
        project_domains = [domain for domain, score in analysis['domain_scores'].items() if score > 0.3]
        
        for use_case in use_cases:
            use_case_lower = use_case.lower()
            for domain in project_domains:
                domain_keywords = self.domain_keywords[domain]
                if any(keyword in use_case_lower for keyword in domain_keywords):
                    alignment_score += 0.2
        
        return min(alignment_score, 1.0)
    
    def _determine_agent_priority(self, confidence_score: float, agent: Dict, analysis: Dict) -> int:
        """Determine agent priority level (1=Critical, 2=Important, 3=Helpful)"""
        if confidence_score >= 0.7:
            return 1  # Critical
        elif confidence_score >= 0.4:
            return 2  # Important
        else:
            return 3  # Helpful
    
    def _estimate_agent_monthly_cost(self, agent: Dict, analysis: Dict) -> float:
        """Estimate monthly cost for using this agent"""
        pricing = agent.get('pricing', {})
        
        # Extract individual monthly cost if available
        if 'individual_per_month' in pricing and pricing['individual_per_month'] is not None:
            return float(pricing['individual_per_month'])
        
        # For token-based pricing, estimate based on usage
        if 'prompt_tokens_per_1K' in pricing:
            # Estimate tokens per month based on project complexity
            complexity_multiplier = {'low': 100000, 'medium': 300000, 'high': 800000}
            estimated_tokens = complexity_multiplier.get(analysis['complexity'], 300000)
            
            prompt_cost = pricing.get('prompt_tokens_per_1K', 0) * (estimated_tokens / 1000)
            completion_cost = pricing.get('completion_tokens_per_1K', 0) * (estimated_tokens / 1000)
            
            return prompt_cost + completion_cost
        
        # Default estimates for agents without clear pricing
        return 0.0
    
    def _generate_agent_reasoning(self, agent: Dict, analysis: Dict, relevant_domains: List[str], score: float) -> str:
        """Generate human-readable reasoning for agent recommendation"""
        agent_name = agent['name']
        complexity = analysis['complexity']
        
        if score >= 0.7:
            if relevant_domains:
                return f"Excellent match for {', '.join(relevant_domains)} development with {complexity} complexity"
            else:
                return f"Strong capabilities align well with {complexity} complexity projects"
        elif score >= 0.4:
            return f"Good fit for {complexity} complexity projects, particularly for {', '.join(relevant_domains[:2])}"
        else:
            return f"Could provide value for specific aspects of this {complexity} complexity project"
    
    def match_human_experts(self, analysis: Dict) -> List[Tuple[HumanExpert, float, str]]:
        """Match human experts to project requirements"""
        expert_matches = []
        
        for expert in self.human_experts:
            confidence, reasoning = self._calculate_expert_match_score(expert, analysis)
            
            if confidence > 0.2:  # Only include meaningful matches
                expert_matches.append((expert, confidence, reasoning))
        
        # Sort by confidence score
        expert_matches.sort(key=lambda x: x[1], reverse=True)
        
        return expert_matches
    
    def _calculate_expert_match_score(self, expert: HumanExpert, analysis: Dict) -> Tuple[float, str]:
        """Calculate expert match score and reasoning"""
        
        # Calculate domain alignment
        domain_score = 0.0
        matched_domains = []
        
        for domain in expert.experience_domains:
            if domain in analysis['domain_scores'] and analysis['domain_scores'][domain] > 0.3:
                domain_score += analysis['domain_scores'][domain]
                matched_domains.append(domain)
        
        # Normalize domain score
        domain_score = min(domain_score / len(expert.experience_domains), 1.0)
        
        # Calculate complexity alignment
        complexity_score = self._calculate_expert_complexity_match(expert, analysis['complexity'])
        
        # Calculate team size factor
        team_size_score = self._calculate_team_size_factor(expert, analysis['team_size'])
        
        # Weighted final score
        final_score = (
            domain_score * 0.5 +
            complexity_score * 0.3 +
            team_size_score * 0.2
        )
        
        # Generate reasoning
        reasoning = self._generate_expert_reasoning(expert, matched_domains, analysis['complexity'])
        
        return final_score, reasoning
    
    def _calculate_expert_complexity_match(self, expert: HumanExpert, complexity: str) -> float:
        """Calculate how well expert matches project complexity"""
        # Senior roles are better for high complexity
        if 'senior' in expert.role.lower() or 'lead' in expert.role.lower():
            return {'low': 0.6, 'medium': 0.8, 'high': 1.0}[complexity]
        else:
            return {'low': 1.0, 'medium': 0.8, 'high': 0.6}[complexity]
    
    def _calculate_team_size_factor(self, expert: HumanExpert, team_size: Dict[str, int]) -> float:
        """Calculate if expert is needed based on team size requirements"""
        role_mapping = {
            'Senior Full-Stack Developer': 'developers',
            'UI/UX Designer': 'designers',
            'DevOps Engineer': 'devops',
            'QA Engineer': 'qa',
            'Data Scientist': 'data_specialists',
            'Technical Project Manager': 'project_managers'
        }
        
        team_key = role_mapping.get(expert.role)
        if team_key and team_key in team_size:
            return 1.0 if team_size[team_key] > 0 else 0.3
        
        return 0.5  # Default if not mapped
    
    def _generate_expert_reasoning(self, expert: HumanExpert, matched_domains: List[str], complexity: str) -> str:
        """Generate reasoning for expert recommendation"""
        if matched_domains:
            domain_text = f"Strong expertise in {', '.join(matched_domains)}"
        else:
            domain_text = "Valuable general expertise"
        
        complexity_text = f"Well-suited for {complexity} complexity projects"
        
        return f"{domain_text}. {complexity_text}."
    
    def generate_comprehensive_recommendation(self, project_brief: str, agent_catalog: List[Dict]) -> Dict[str, Any]:
        """
        Generate complete recommendation including agents, experts, costs, and strategy
        """
        # Analyze project requirements
        analysis = self.extract_keywords_and_analyze(project_brief)
        
        # Match AI agents
        agent_matches = self.match_agents_to_requirements(analysis, agent_catalog)
        
        # Match human experts
        expert_matches = self.match_human_experts(analysis)
        
        # Generate team composition recommendations
        team_composition = self._generate_team_composition(agent_matches, expert_matches, analysis)
        
        # Estimate costs
        cost_estimates = self._estimate_project_costs(analysis, agent_matches, expert_matches)
        
        # Generate implementation strategy
        implementation_strategy = self._generate_implementation_strategy(
            analysis, agent_matches, expert_matches
        )
        
        # Identify success factors
        success_factors = self._identify_success_factors(analysis, agent_matches, expert_matches)
        
        return {
            'project_analysis': analysis,
            'recommended_agents': agent_matches,
            'recommended_experts': expert_matches,
            'team_composition': team_composition,
            'cost_estimates': cost_estimates,
            'implementation_strategy': implementation_strategy,
            'success_factors': success_factors
        }
    
    def _generate_team_composition(self, agent_matches: List[MatchResult], 
                                  expert_matches: List[Tuple[HumanExpert, float, str]], 
                                  analysis: Dict) -> Dict[str, Any]:
        """Generate optimal team composition recommendations"""
        
        # Categorize agents by function
        agent_categories = {
            'coding_assistants': [],
            'design_tools': [],
            'testing_tools': [],
            'documentation_tools': [],
            'devops_tools': []
        }
        
        for match in agent_matches[:10]:  # Top 10 agents
            agent_name = match.agent_name.lower()
            
            if any(term in agent_name for term in ['copilot', 'codewhisperer', 'tabnine', 'codeium']):
                agent_categories['coding_assistants'].append(match)
            elif any(term in agent_name for term in ['galileo', 'uizard', 'design']):
                agent_categories['design_tools'].append(match)
            elif any(term in agent_name for term in ['test', 'diffblue', 'cover']):
                agent_categories['testing_tools'].append(match)
            elif any(term in agent_name for term in ['docs', 'mintlify', 'documentation']):
                agent_categories['documentation_tools'].append(match)
            elif any(term in agent_name for term in ['aws', 'cloud', 'devops']):
                agent_categories['devops_tools'].append(match)
            else:
                agent_categories['coding_assistants'].append(match)  # Default category
        
        # Select top experts by role
        expert_roles = {}
        for expert, confidence, reasoning in expert_matches[:6]:  # Top 6 experts
            if expert.role not in expert_roles or confidence > expert_roles[expert.role][1]:
                expert_roles[expert.role] = (expert, confidence, reasoning)
        
        return {
            'ai_agents': agent_categories,
            'human_experts': expert_roles,
            'team_size_recommendation': analysis['team_size'],
            'collaboration_suggestions': self._generate_collaboration_suggestions(agent_matches, expert_matches)
        }
    
    def _generate_collaboration_suggestions(self, agent_matches: List[MatchResult],
                                          expert_matches: List[Tuple[HumanExpert, float, str]]) -> List[str]:
        """Generate suggestions for AI-human collaboration"""
        suggestions = []
        
        # Find coding assistants and developers
        coding_agents = [m for m in agent_matches if 'copilot' in m.agent_name.lower() or 'code' in m.agent_name.lower()]
        developers = [e for e, _, _ in expert_matches if 'developer' in e.role.lower()]
        
        if coding_agents and developers:
            suggestions.append(f"Pair {coding_agents[0].agent_name} with {developers[0].role} for accelerated development")
        
        # Find design tools and designers
        design_agents = [m for m in agent_matches if any(term in m.agent_name.lower() for term in ['galileo', 'uizard', 'design'])]
        designers = [e for e, _, _ in expert_matches if 'designer' in e.role.lower()]
        
        if design_agents and designers:
            suggestions.append(f"Use {design_agents[0].agent_name} with {designers[0].role} for rapid prototyping")
        
        return suggestions
    
    def _estimate_project_costs(self, analysis: Dict, agent_matches: List[MatchResult],
                               expert_matches: List[Tuple[HumanExpert, float, str]]) -> Dict[str, float]:
        """Estimate total project costs"""
        
        # Calculate AI tools monthly cost (top 5 agents)
        ai_monthly_cost = sum(match.estimated_monthly_cost for match in agent_matches[:5])
        
        # Calculate human expert monthly cost (top 3 experts)
        expert_monthly_cost = 0.0
        for expert, confidence, _ in expert_matches[:3]:
            # Extract hourly rate (take middle of range)
            rate_range = expert.hourly_rate_range
            rate_match = re.search(r'\$(\d+)-(\d+)', rate_range)
            if rate_match:
                low_rate = int(rate_match.group(1))
                high_rate = int(rate_match.group(2))
                avg_rate = (low_rate + high_rate) / 2
                
                # Estimate hours per month based on confidence and project complexity
                complexity_hours = {'low': 40, 'medium': 80, 'high': 120}
                monthly_hours = complexity_hours[analysis['complexity']] * confidence
                
                expert_monthly_cost += avg_rate * monthly_hours
        
        # Estimate project duration
        complexity_duration = {'low': 2, 'medium': 4, 'high': 8}
        base_duration = complexity_duration[analysis['complexity']]
        
        # Adjust duration based on team size and urgency
        team_factor = max(0.5, 1.0 / max(1, analysis['team_size']['developers']))
        urgency_factor = {'urgent': 0.8, 'normal': 1.0, 'flexible': 1.2}[analysis['urgency']]
        
        estimated_duration = base_duration * team_factor * urgency_factor
        
        # Calculate total cost
        total_cost = (ai_monthly_cost + expert_monthly_cost) * estimated_duration
        
        return {
            'ai_tools_monthly': round(ai_monthly_cost, 2),
            'human_experts_monthly': round(expert_monthly_cost, 2),
            'estimated_project_duration_months': round(estimated_duration, 1),
            'total_estimated_cost': round(total_cost, 2)
        }
    
    def _generate_implementation_strategy(self, analysis: Dict, agent_matches: List[MatchResult],
                                        expert_matches: List[Tuple[HumanExpert, float, str]]) -> Dict[str, Any]:
        """Generate phased implementation strategy"""
        
        complexity = analysis['complexity']
        domains = [d for d, score in analysis['domain_scores'].items() if score > 0.3]
        
        phases = []
        
        # Phase 1: Setup and Planning
        phase1_agents = [m.agent_name for m in agent_matches if 'gpt' in m.agent_name.lower() or 'claude' in m.agent_name.lower()][:2]
        phase1_experts = ['Technical Project Manager', 'Senior Full-Stack Developer']
        
        phases.append({
            'phase': 'Setup & Architecture',
            'duration': '2-3 weeks',
            'key_agents': phase1_agents,
            'key_experts': phase1_experts,
            'deliverables': ['Technical specification', 'Architecture design', 'Development environment setup']
        })
        
        # Phase 2: Core Development
        phase2_agents = [m.agent_name for m in agent_matches if any(term in m.agent_name.lower() for term in ['copilot', 'codewhisperer'])][:2]
        phase2_experts = ['Senior Full-Stack Developer']
        if 'design' in domains:
            phase2_experts.append('UI/UX Designer')
        
        duration = {'low': '4-6 weeks', 'medium': '8-12 weeks', 'high': '16-20 weeks'}[complexity]
        phases.append({
            'phase': 'Core Development',
            'duration': duration,
            'key_agents': phase2_agents,
            'key_experts': phase2_experts,
            'deliverables': ['MVP implementation', 'Core features', 'Initial testing']
        })
        
        # Phase 3: Testing and Deployment
        phase3_agents = []
        phase3_experts = ['QA Engineer']
        if 'cloud' in domains:
            phase3_experts.append('DevOps Engineer')
        
        phases.append({
            'phase': 'Testing & Deployment',
            'duration': '3-4 weeks',
            'key_agents': phase3_agents,
            'key_experts': phase3_experts,
            'deliverables': ['Comprehensive testing', 'Production deployment', 'Documentation']
        })
        
        # Generate risk mitigation strategies
        risk_mitigation = self._generate_risk_mitigation(analysis, complexity)
        
        return {
            'phases': phases,
            'total_estimated_timeline': f"{len(phases) * 4}-{len(phases) * 8} weeks",
            'risk_mitigation': risk_mitigation,
            'success_metrics': self._generate_success_metrics(analysis)
        }
    
    def _generate_risk_mitigation(self, analysis: Dict, complexity: str) -> List[str]:
        """Generate risk mitigation strategies"""
        risks = []
        
        if complexity == 'high':
            risks.append("Break down complex features into smaller, manageable components")
            risks.append("Implement frequent code reviews and pair programming sessions")
        
        if analysis['urgency'] == 'urgent':
            risks.append("Prioritize core features and defer nice-to-have functionality")
            risks.append("Increase team communication frequency and use daily standups")
        
        if len([d for d, s in analysis['domain_scores'].items() if s > 0.4]) > 3:
            risks.append("Consider hiring specialists for complex domains early in the project")
        
        risks.append("Maintain regular stakeholder communication to prevent scope creep")
        risks.append("Set up monitoring and alerting systems from the beginning")
        
        return risks
    
    def _generate_success_metrics(self, analysis: Dict) -> List[str]:
        """Generate success metrics for the project"""
        metrics = [
            "Code quality metrics (test coverage > 80%)",
            "Performance benchmarks (page load < 2 seconds)",
            "User acceptance criteria completion",
            "Technical debt measurements"
        ]
        
        if 'cloud' in [d for d, s in analysis['domain_scores'].items() if s > 0.4]:
            metrics.append("Infrastructure reliability (99.9% uptime)")
        
        if 'frontend' in [d for d, s in analysis['domain_scores'].items() if s > 0.4]:
            metrics.append("User experience metrics (accessibility compliance)")
        
        return metrics
    
    def _identify_success_factors(self, analysis: Dict, agent_matches: List[MatchResult],
                                 expert_matches: List[Tuple[HumanExpert, float, str]]) -> List[str]:
        """Identify key success factors for the project"""
        factors = []
        
        # AI tool factors
        if agent_matches:
            top_agent = agent_matches[0]
            factors.append(f"Leverage {top_agent.agent_name} for {top_agent.reasoning.lower()}")
        
        # Human expertise factors
        if expert_matches:
            top_expert = expert_matches[0][0]
            factors.append(f"Engage {top_expert.role} early for technical guidance and architecture decisions")
        
        # Complexity factors
        if analysis['complexity'] == 'high':
            factors.append("Implement robust testing strategy and CI/CD pipeline from the start")
            factors.append("Plan for scalability and performance optimization")
        
        # Domain-specific factors
        domains = [d for d, score in analysis['domain_scores'].items() if score > 0.4]
        if 'frontend' in domains:
            factors.append("Focus on responsive design and user experience optimization")
        if 'backend' in domains:
            factors.append("Ensure robust API design and data security measures")
        if 'cloud' in domains:
            factors.append("Implement infrastructure as code and monitoring from day one")
        
        # Timeline factors
        if analysis['urgency'] == 'urgent':
            factors.append("Prioritize MVP features and plan iterative releases")
        
        return factors
