"""Base agent class with Groq real-time inference capabilities"""

import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from ..models.farm_context import FarmContext
from ..models.agent_response import AgentResponse, Confidence, ResponseType
from ..services.groq_service import GroqService
from ..config import settings
from ..exceptions import AgentException, GroqAPIException


class BaseAgent(ABC):
    """Base class for all AgriGuard agents"""
    
    def __init__(self, groq_service: GroqService, agent_name: str):
        self.groq_service = groq_service
        self.agent_name = agent_name
        self.agent_version = "1.0.0"
        self.response_cache = {}
        
    @abstractmethod
    def get_system_prompt(self, context: FarmContext) -> str:
        """Get agent-specific system prompt"""
        pass
    
    @abstractmethod
    def get_agent_capabilities(self) -> List[str]:
        """Get list of agent capabilities"""
        pass
    
    async def process_query(self, query: str, context: FarmContext) -> AgentResponse:
        """Process query with real-time Groq inference"""
        
        start_time = time.time()
        
        try:
            # Build system prompt with context
            system_prompt = self.get_system_prompt(context)
            
            # Prepare user prompt
            user_prompt = self._build_user_prompt(query, context)
            
            # Get Groq inference
            groq_start = time.time()
            raw_response = await self.groq_service.get_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800 if context.is_emergency() else 1500,
                temperature=0.1 if context.is_emergency() else 0.3
            )
            groq_time = (time.time() - groq_start) * 1000
            
            # Parse and structure response
            structured_response = await self._parse_response(raw_response, context)
            
            # Create AgentResponse object
            response = AgentResponse(
                agent_name=self.agent_name,
                agent_version=self.agent_version,
                response_type=self._determine_response_type(structured_response, context),
                summary=structured_response.get('summary', '')[:200],
                detailed_response=structured_response.get('detailed_response', ''),
                confidence_score=self._calculate_confidence(structured_response),
                groq_inference_time_ms=groq_time,
                context_used=context.to_context_summary()
            )
            
            # Add structured data
            self._populate_response_data(response, structured_response, context)
            
            return response
            
        except Exception as e:
            raise AgentException(self.agent_name, f"Query processing failed: {str(e)}")
    
    def _build_user_prompt(self, query: str, context: FarmContext) -> str:
        """Build comprehensive user prompt with context"""
        
        context_summary = context.to_context_summary()
        primary_crop = context.get_primary_crop()
        
        prompt = f"""
FARMER QUERY: {query}

FARM CONTEXT:
- Farmer: {context_summary['farmer_name']}
- Location: {context_summary['location']}
- Primary Crop: {primary_crop.crop_type.value} ({primary_crop.variety or 'Standard variety'})
- Growth Stage: {primary_crop.growth_stage}
- Farm Size: {context_summary['farm_size']} acres
- Current Season: {context_summary['season']}
- Emergency Level: {context_summary['emergency_level']}/5
- Farming Method: {context_summary['farming_method']}

CURRENT CONDITIONS:
- Planting Date: {primary_crop.planting_date.strftime('%Y-%m-%d')}
- Area Under Crop: {primary_crop.area_planted} acres
"""

        # Add soil data if available
        if context.soil_data:
            prompt += f"""
SOIL CONDITIONS:
- Soil Type: {context.soil_data.soil_type}
- pH Level: {context.soil_data.ph_level}
- Moisture: {context.soil_data.moisture_level}%
- NPK: N:{context.soil_data.nitrogen}, P:{context.soil_data.phosphorus}, K:{context.soil_data.potassium}
"""

        # Add emergency context
        if context.is_emergency():
            prompt += f"""
⚠️ EMERGENCY SITUATION - PRIORITY RESPONSE REQUIRED
Emergency Level: {context.emergency_level}/5
Respond with immediate, actionable advice.
"""

        return prompt
    
    async def _parse_response(self, raw_response: str, context: FarmContext) -> Dict[str, Any]:
        """Parse raw Groq response into structured format"""
        
        try:
            # Try to parse as JSON first
            if raw_response.strip().startswith('{'):
                return json.loads(raw_response)
        except json.JSONDecodeError:
            pass
        
        # Fallback: Structure plain text response
        lines = raw_response.strip().split('\n')
        summary = lines[0] if lines else "Agricultural advice provided"
        
        return {
            "summary": summary[:200],
            "detailed_response": raw_response,
            "confidence": "medium",
            "action_items": self._extract_action_items(raw_response),
            "emergency_indicators": self._detect_emergency_indicators(raw_response)
        }
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from response text"""
        actions = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for action indicators
            if any(indicator in line.lower() for indicator in 
                   ['immediately', 'urgent', 'action:', 'do:', 'apply:', 'spray:', 'contact:']):
                # Clean up the action
                action = line.replace('•', '').replace('-', '').strip()
                if action and len(action) > 10:
                    actions.append(action)
        
        return actions[:5]  # Limit to 5 actions
    
    def _detect_emergency_indicators(self, text: str) -> Dict[str, Any]:
        """Detect emergency indicators in response"""
        text_lower = text.lower()
        
        emergency_keywords = ['urgent', 'immediate', 'emergency', 'critical', 'severe', 
                            'outbreak', 'spreading', 'dying', 'damage', 'loss']
        
        emergency_score = sum(1 for keyword in emergency_keywords if keyword in text_lower)
        
        return {
            "emergency_detected": emergency_score > 2,
            "emergency_score": min(5, emergency_score),
            "keywords_found": [kw for kw in emergency_keywords if kw in text_lower]
        }
    
    def _determine_response_type(self, structured_response: Dict, context: FarmContext) -> ResponseType:
        """Determine appropriate response type"""
        
        if context.is_emergency() or structured_response.get('emergency_indicators', {}).get('emergency_detected'):
            return ResponseType.EMERGENCY
        
        if structured_response.get('emergency_indicators', {}).get('emergency_score', 0) > 1:
            return ResponseType.WARNING
        
        if structured_response.get('action_items'):
            return ResponseType.ACTION_REQUIRED
        
        return ResponseType.ADVISORY
    
    def _calculate_confidence(self, structured_response: Dict) -> Confidence:
        """Calculate confidence score for response"""
        
        # Base confidence on response completeness and specificity
        confidence_score = 0.7  # Start with medium
        
        if structured_response.get('detailed_response') and len(structured_response['detailed_response']) > 100:
            confidence_score += 0.1
        
        if structured_response.get('action_items'):
            confidence_score += 0.1
        
        if structured_response.get('confidence') == 'high':
            confidence_score += 0.1
        
        # Convert to enum
        if confidence_score >= 0.9:
            return Confidence.VERY_HIGH
        elif confidence_score >= 0.75:
            return Confidence.HIGH
        elif confidence_score >= 0.5:
            return Confidence.MEDIUM
        elif confidence_score >= 0.25:
            return Confidence.LOW
        else:
            return Confidence.VERY_LOW
    
    def _populate_response_data(self, response: AgentResponse, structured_data: Dict, context: FarmContext):
        """Populate additional response data"""
        
        # Add action items
        for action_text in structured_data.get('action_items', []):
            response.add_action_item(action_text, priority=4 if context.is_emergency() else 3)
        
        # Set emergency status
        emergency_indicators = structured_data.get('emergency_indicators', {})
        if emergency_indicators.get('emergency_detected') or context.is_emergency():
            response.mark_as_emergency(
                level=max(context.emergency_level, emergency_indicators.get('emergency_score', 4)),
                escalate=True
            )
        
        # Add immediate actions for emergencies
        if response.is_emergency and structured_data.get('action_items'):
            response.immediate_actions = structured_data['action_items'][:3]
        
        # Set data sources
        response.data_sources = [f"groq_{settings.groq_model}", "farm_context", f"{self.agent_name}_knowledge"]
