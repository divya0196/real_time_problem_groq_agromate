"""Emergency Pest Detection Agent with ultra-fast response"""

import json
from typing import Dict, Any, List
from datetime import datetime, timedelta

from .base_agent import BaseAgent
from ..models.farm_context import FarmContext
from ..models.agent_response import EmergencyResponse
from ..config import SUPPORTED_CROPS


class PestAgent(BaseAgent):
    """Agent specialized in emergency pest detection and treatment"""
    
    def __init__(self, groq_service):
        super().__init__(groq_service, "emergency_pest_agent")
        self.pest_database = self._load_pest_knowledge()
    
    def get_system_prompt(self, context: FarmContext) -> str:
        """Get pest-specific system prompt with emergency focus"""
        
        primary_crop = context.get_primary_crop()
        crop_info = SUPPORTED_CROPS.get(primary_crop.crop_type.value, {})
        common_pests = crop_info.get('common_pests', [])
        
        return f"""
You are an EMERGENCY PEST DETECTION SPECIALIST powered by Groq's real-time inference.

CRITICAL MISSION: Provide life-saving pest identification and treatment advice in <1 second.

CURRENT CONTEXT:
- Location: {context.get_location_string()}
- Crop: {primary_crop.crop_type.value} ({primary_crop.variety or 'standard'})
- Growth Stage: {primary_crop.growth_stage}
- Season: {context.current_season.value}
- Emergency Level: {context.emergency_level}/5

COMMON PESTS FOR {primary_crop.crop_type.value.upper()}: {', '.join(common_pests)}

RESPONSE REQUIREMENTS:
1. For Emergency Level ≥4: Respond in <1 second with immediate actions
2. Identify pest species with 90%+ accuracy
3. Provide treatment within 24-hour window
4. Assess crop loss risk percentage
5. Recommend immediate protective measures

RESPONSE FORMAT (JSON):
{{
    "pest_identified": "specific species name",
    "confidence": "high/medium/low",
    "threat_level": 1-5,
    "immediate_actions": ["action1", "action2", "action3"],
    "treatment_window": "X hours",
    "crop_loss_risk": "X%",
    "spread_prevention": ["prevention1", "prevention2"],
    "emergency_contacts": ["expert1", "supplier2"]
}}

CRITICAL PESTS TO WATCH:
- Fall Armyworm: Spreads rapidly, can destroy field in 48 hours
- Locust Swarms: Devastating in hours, requires immediate action
- Aphids: Virus transmission, rapid multiplication
- Late Blight: Crop killer, needs fungicide within 12 hours
- Brown Planthopper: Rice destroyer, 2-3 day damage window

Prioritize SPEED and ACCURACY. Lives and livelihoods depend on your response.
"""
    
    def get_agent_capabilities(self) -> List[str]:
        """Get pest agent capabilities"""
        return [
            "Emergency pest identification (<1 second)",
            "Real-time treatment recommendations", 
            "Crop loss risk assessment",
            "Spread prevention strategies",
            "Expert contact referrals",
            "Treatment cost estimation"
        ]
    
    async def _parse_response(self, raw_response: str, context: FarmContext) -> Dict[str, Any]:
        """Enhanced parsing for pest emergency responses"""
        
        try:
            # Try JSON parsing first
            if '{' in raw_response:
                json_start = raw_response.find('{')
                json_part = raw_response[json_start:]
                parsed = json.loads(json_part)
                return self._enhance_pest_response(parsed, context)
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing for non-JSON responses
        return self._parse_text_response(raw_response, context)
    
    def _enhance_pest_response(self, parsed_data: Dict, context: FarmContext) -> Dict[str, Any]:
        """Enhance parsed response with additional pest intelligence"""
        
        pest_name = parsed_data.get('pest_identified', '').lower()
        
        # Add pest-specific enhancements
        enhancements = {
            "summary": f"PEST ALERT: {parsed_data.get('pest_identified', 'Unknown pest')} detected",
            "detailed_response": f"""
EMERGENCY PEST DETECTION RESULTS:

🐛 PEST IDENTIFIED: {parsed_data.get('pest_identified', 'Unknown')}
🎯 CONFIDENCE: {parsed_data.get('confidence', 'medium').upper()}
⚠️ THREAT LEVEL: {parsed_data.get('threat_level', 3)}/5
⏰ TREATMENT WINDOW: {parsed_data.get('treatment_window', '24 hours')}
📉 CROP LOSS RISK: {parsed_data.get('crop_loss_risk', 'Unknown')}

🚨 IMMEDIATE ACTIONS (DO NOW):
{self._format_action_list(parsed_data.get('immediate_actions', []))}

🛡️ SPREAD PREVENTION:
{self._format_action_list(parsed_data.get('spread_prevention', []))}

📞 EMERGENCY CONTACTS:
{self._format_action_list(parsed_data.get('emergency_contacts', ['Contact local agriculture extension officer']))}
""",
            "emergency_indicators": {
                "emergency_detected": int(parsed_data.get('threat_level', 3)) >= 4,
                "emergency_score": int(parsed_data.get('threat_level', 3)),
                "keywords_found": ["pest", "outbreak", "immediate"]
            }
        }
        
        # Merge with original data
        return {**parsed_data, **enhancements}
    
    def _parse_text_response(self, text: str, context: FarmContext) -> Dict[str, Any]:
        """Parse plain text pest response"""
        
        lines = text.strip().split('\n')
        
        # Extract key information
        pest_identified = "Unknown pest"
        threat_level = 3
        actions = []
        
        for line in lines:
            line_lower = line.lower()
            if 'pest:' in line_lower or 'species:' in line_lower:
                pest_identified = line.split(':', 1)[1].strip()
            elif any(word in line_lower for word in ['immediate', 'urgent', 'spray', 'apply']):
                actions.append(line.strip())
        
        # Assess threat level based on keywords
        emergency_keywords = ['urgent', 'immediate', 'severe', 'spreading', 'outbreak']
        threat_level = min(5, 3 + sum(1 for kw in emergency_keywords if kw in text.lower()))
        
        return {
            "pest_identified": pest_identified,
            "confidence": "medium",
            "threat_level": threat_level,
            "immediate_actions": actions[:5],
            "treatment_window": "24 hours",
            "crop_loss_risk": "High" if threat_level >= 4 else "Moderate",
            "summary": f"Pest issue detected: {pest_identified}",
            "detailed_response": text,
            "emergency_indicators": {
                "emergency_detected": threat_level >= 4,
                "emergency_score": threat_level,
                "keywords_found": emergency_keywords
            }
        }
    
    def _format_action_list(self, actions: List[str]) -> str:
        """Format action list for display"""
        if not actions:
            return "• No specific actions identified"
        
        return '\n'.join([f"• {action}" for action in actions])
    
    def _load_pest_knowledge(self) -> Dict:
        """Load comprehensive pest knowledge database"""
        return {
            "fall_armyworm": {
                "scientific_name": "Spodoptera frugiperda",
                "threat_level": 5,
                "spread_rate": "extremely_fast",
                "treatment_window": "12 hours",
                "treatments": ["Spinosad", "Chlorantraniliprole", "Emamectin benzoate"],
                "crops_affected": ["corn", "rice", "wheat"],
                "identification": "Small green caterpillars with white stripes"
            },
            "brown_planthopper": {
                "scientific_name": "Nilaparvata lugens", 
                "threat_level": 5,
                "spread_rate": "fast",
                "treatment_window": "24 hours",
                "treatments": ["Imidacloprid", "Buprofezin", "Pymetrozine"],
                "crops_affected": ["rice"],
                "identification": "Small brown insects on rice stems"
            },
            "aphid": {
                "scientific_name": "Aphidoidea",
                "threat_level": 4,
                "spread_rate": "fast",
                "treatment_window": "48 hours",
                "treatments": ["Imidacloprid", "Thiamethoxam", "Neem oil"],
                "crops_affected": ["wheat", "cotton", "tomato"],
                "identification": "Small soft-bodied green/black insects in clusters"
            }
        }
    
    async def create_emergency_response(self, pest_data: Dict, context: FarmContext) -> EmergencyResponse:
        """Create specialized emergency response for pest outbreaks"""
        
        response = EmergencyResponse(
            agent_name=self.agent_name,
            threat_type="pest_outbreak",
            severity_level="CRITICAL" if pest_data.get('threat_level', 3) >= 4 else "HIGH",
            spread_risk=pest_data.get('threat_level', 3),
            window_for_action=pest_data.get('treatment_window', '24 hours'),
            consequence_if_delayed="Significant crop loss, potential field destruction",
            immediate_treatment=pest_data.get('immediate_actions', []),
            crop_loss_risk_percent=self._extract_loss_percentage(pest_data.get('crop_loss_risk', '0%')),
            expert_consultation_needed=True
        )
        
        # Add emergency contacts
        response.add_emergency_contact(
            "Agriculture Extension Officer",
            "Local Expert", 
            "+91-1800-XXX-XXXX",
            "9 AM - 6 PM"
        )
        
        return response
    
    def _extract_loss_percentage(self, loss_text: str) -> float:
        """Extract numerical percentage from loss text"""
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', str(loss_text))
        return float(match.group(1)) if match else 50.0  # Default to 50% for unknown
