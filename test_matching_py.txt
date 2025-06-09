"""
Unit tests for the matching system core functionality
"""

import unittest
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matching_system import ProjectRequirementAnalyzer, MatchResult, HumanExpert

class TestProjectRequirementAnalyzer(unittest.TestCase):
    """Test cases for the ProjectRequirementAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ProjectRequirementAnalyzer()
        self.sample_catalog = [
            {
                "name": "GitHub Copilot",
                "provider": "GitHub",
                "type": "AI pair-programmer",
                "capabilities": ["inline code suggestions", "frontend development"],
                "limitations": ["limited context"],
                "integration": ["VS Code"],
                "ideal_use_cases": ["day-to-day coding"],
                "pricing": {"individual_per_month": 10}
            },
            {
                "name": "OpenAI GPT-4",
                "provider": "OpenAI", 
                "type": "General-purpose LLM",
                "capabilities": ["complex code generation", "documentation"],
                "limitations": ["higher cost"],
                "integration": ["OpenAI API"],
                "ideal_use_cases": ["complex backend logic"],
                "pricing": {"prompt_tokens_per_1K": 0.03}
            }
        ]
    
    def test_extract_keywords_and_analyze(self):
        """Test keyword extraction and analysis"""
        brief = "Building a React frontend with Python backend, deploy on AWS"
        
        analysis = self.analyzer.extract_keywords_and_analyze(brief)
        
        # Check structure
        self.assertIn('domain_scores', analysis)
        self.assertIn('complexity', analysis)
        self.assertIn('urgency', analysis)
        self.assertIn('technologies', analysis)
        
        # Check domain detection
        self.assertGreater(analysis['domain_scores']['frontend'], 0)
        self.assertGreater(analysis['domain_scores']['backend'], 0)
        self.assertGreater(analysis['domain_scores']['cloud'], 0)
        
        # Check technology extraction
        self.assertIn('react', analysis['technologies'])
        self.assertIn('python', analysis['technologies'])
        self.assertIn('aws', analysis['technologies'])
    
    def test_complexity_determination(self):
        """Test complexity level determination"""
        simple_brief = "Simple React app with basic features"
        complex_brief = "Enterprise-scale distributed microservices with complex business logic"
        
        simple_analysis = self.analyzer.extract_keywords_and_analyze(simple_brief)
        complex_analysis = self.analyzer.extract_keywords_and_analyze(complex_brief)
        
        # Simple project should be detected as low complexity
        self.assertIn(simple_analysis['complexity'], ['low', 'medium'])
        
        # Complex project should be detected as high complexity
        self.assertEqual(complex_analysis['complexity'], 'high')
    
    def test_urgency_assessment(self):
        """Test urgency level assessment"""
        urgent_brief = "Need this ASAP, very urgent deadline"
        normal_brief = "Standard timeline, no rush"
        
        urgent_analysis = self.analyzer.extract_keywords_and_analyze(urgent_brief)
        normal_analysis = self.analyzer.extract_keywords_and_analyze(normal_brief)
        
        self.assertEqual(urgent_analysis['urgency'], 'urgent')
        self.assertIn(normal_analysis['urgency'], ['normal', 'flexible'])
    
    def test_agent_matching(self):
        """Test agent matching functionality"""
        brief = "React frontend development with code assistance needed"
        analysis = self.analyzer.extract_keywords_and_analyze(brief)
        
        matches = self.analyzer.match_agents_to_requirements(analysis, self.sample_catalog)
        
        # Should return matches
        self.assertGreater(len(matches), 0)
        
        # Check match structure
        for match in matches:
            self.assertIsInstance(match, MatchResult)
            self.assertIsNotNone(match.agent_name)
            self.assertIsInstance(match.confidence_score, float)
            self.assertGreaterEqual(match.confidence_score, 0)
            self.assertLessEqual(match.confidence_score, 1)
            self.assertIn(match.priority, [1, 2, 3])
        
        # GitHub Copilot should score well for frontend development
        copilot_match = next((m for m in matches if m.agent_name == "GitHub Copilot"), None)
        self.assertIsNotNone(copilot_match)
        self.assertGreater(copilot_match.confidence_score, 0.3)
    
    def test_human_expert_matching(self):
        """Test human expert matching"""
        brief = "Complex full-stack application requiring senior oversight"
        analysis = self.analyzer.extract_keywords_and_analyze(brief)
        
        expert_matches = self.analyzer.match_human_experts(analysis)
        
        # Should return expert matches
        self.assertGreater(len(expert_matches), 0)
        
        # Check match structure
        for expert, confidence, reasoning in expert_matches:
            self.assertIsInstance(expert, HumanExpert)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)
            self.assertIsInstance(reasoning, str)
            self.assertGreater(len(reasoning), 0)
    
    def test_comprehensive_recommendation(self):
        """Test comprehensive recommendation generation"""
        brief = "Building e-commerce platform with React and Python, need testing and docs"
        
        recommendation = self.analyzer.generate_comprehensive_recommendation(brief, self.sample_catalog)
        
        # Check all required sections
        required_keys = [
            'project_analysis', 'recommended_agents', 'recommended_experts',
            'team_composition', 'cost_estimates', 'implementation_strategy',
            'success_factors'
        ]
        
        for key in required_keys:
            self.assertIn(key, recommendation)
        
        # Check data types and basic validation
        self.assertIsInstance(recommendation['project_analysis'], dict)
        self.assertIsInstance(recommendation['recommended_agents'], list)
        self.assertIsInstance(recommendation['recommended_experts'], list)
        self.assertIsInstance(recommendation['cost_estimates'], dict)
        self.assertIsInstance(recommendation['success_factors'], list)
    
    def test_cost_estimation(self):
        """Test cost estimation functionality"""
        brief = "Medium complexity web application"
        analysis = self.analyzer.extract_keywords_and_analyze(brief)
        agent_matches = self.analyzer.match_agents_to_requirements(analysis, self.sample_catalog)
        expert_matches = self.analyzer.match_human_experts(analysis)
        
        cost_estimates = self.analyzer._estimate_project_costs(analysis, agent_matches, expert_matches)
        
        # Check cost structure
        required_cost_keys = [
            'ai_tools_monthly', 'human_experts_monthly', 
            'estimated_project_duration_months', 'total_estimated_cost'
        ]
        
        for key in required_cost_keys:
            self.assertIn(key, cost_estimates)
            self.assertIsInstance(cost_estimates[key], (int, float))
            self.assertGreaterEqual(cost_estimates[key], 0)
    
    def test_team_size_estimation(self):
        """Test team size estimation"""
        # Test different complexity levels
        domain_scores = {'frontend': 0.8, 'backend': 0.6, 'cloud': 0.4}
        
        low_team = self.analyzer._estimate_team_size(domain_scores, 'low')
        high_team = self.analyzer._estimate_team_size(domain_scores, 'high')
        
        # High complexity should need more team members
        self.assertGreaterEqual(high_team['developers'], low_team['developers'])
        
        # Should include appropriate specialists for domain scores
        self.assertEqual(low_team['designers'], 1)  # Frontend score > 0.5
        self.assertEqual(low_team['devops'], 1)     # Cloud score > 0.4

class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases"""
    
    def setUp(self):
        self.analyzer = ProjectRequirementAnalyzer()
    
    def test_empty_brief_handling(self):
        """Test handling of empty project brief"""
        analysis = self.analyzer.extract_keywords_and_analyze("")
        
        # Should not crash and return valid structure
        self.assertIsInstance(analysis, dict)
        self.assertIn('complexity', analysis)
        self.assertIn('domain_scores', analysis)
    
    def test_very_long_brief_handling(self):
        """Test handling of very long project brief"""
        long_brief = "React application " * 1000  # Very repetitive long text
        
        analysis = self.analyzer.extract_keywords_and_analyze(long_brief)
        
        # Should handle without issues
        self.assertIsInstance(analysis, dict)
        self.assertGreater(analysis['domain_scores']['frontend'], 0)
    
    def test_invalid_catalog_handling(self):
        """Test handling of invalid agent catalog"""
        invalid_catalog = [
            {"name": "Invalid Agent"},  # Missing required fields
            "not a dict",  # Wrong type
        ]
        
        analysis = self.analyzer.extract_keywords_and_analyze("React app")
        
        # Should not crash with invalid catalog
        matches = self.analyzer.match_agents_to_requirements(analysis, invalid_catalog)
        self.assertIsInstance(matches, list)
    
    def test_special_characters_in_brief(self):
        """Test handling of special characters and non-English text"""
        special_brief = "React app with émojis 🚀 and spëcial chars @#$%"
        
        analysis = self.analyzer.extract_keywords_and_analyze(special_brief)
        
        # Should extract basic keywords despite special characters
        self.assertIn('react', analysis['technologies'])
        self.assertGreater(analysis['domain_scores']['frontend'], 0)

class TestMatchingAccuracy(unittest.TestCase):
    """Test matching accuracy with known scenarios"""
    
    def setUp(self):
        self.analyzer = ProjectRequirementAnalyzer()
        self.full_catalog = [
            {
                "name": "Amazon CodeWhisperer",
                "capabilities": ["AWS integration", "cloud development"],
                "ideal_use_cases": ["serverless applications", "AWS Lambda"]
            },
            {
                "name": "Galileo AI", 
                "capabilities": ["UI design generation", "mockups"],
                "ideal_use_cases": ["design prototyping", "UI creation"]
            },
            {
                "name": "GitHub Copilot",
                "capabilities": ["code completion", "general coding"],
                "ideal_use_cases": ["daily development", "code assistance"]
            }
        ]
    
    def test_aws_project_matching(self):
        """Test that AWS projects recommend AWS-specific tools"""
        aws_brief = "Serverless application on AWS with Lambda and DynamoDB"
        analysis = self.analyzer.extract_keywords_and_analyze(aws_brief)
        matches = self.analyzer.match_agents_to_requirements(analysis, self.full_catalog)
        
        # Should recommend CodeWhisperer for AWS project
        agent_names = [match.agent_name for match in matches]
        self.assertIn("Amazon CodeWhisperer", agent_names)
    
    def test_design_project_matching(self):
        """Test that design projects recommend design tools"""
        design_brief = "Need UI/UX design for mobile app with wireframes and mockups"
        analysis = self.analyzer.extract_keywords_and_analyze(design_brief)
        matches = self.analyzer.match_agents_to_requirements(analysis, self.full_catalog)
        
        # Should recommend design tools
        agent_names = [match.agent_name for match in matches]
        self.assertIn("Galileo AI", agent_names)
    
    def test_general_coding_matching(self):
        """Test that general coding projects recommend general tools"""
        general_brief = "Standard web application with frontend and backend"
        analysis = self.analyzer.extract_keywords_and_analyze(general_brief)
        matches = self.analyzer.match_agents_to_requirements(analysis, self.full_catalog)
        
        # Should recommend general coding tools
        agent_names = [match.agent_name for match in matches]
        self.assertIn("GitHub Copilot", agent_names)

if __name__ == '__main__':
    # Configure test output
    unittest.main(verbosity=2)