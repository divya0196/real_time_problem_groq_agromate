"""
Real-Time AgriGuard Main System
Powered by Groq's ultra-fast inference engine
"""

import asyncio
import os
import time
from typing import Dict, List, Optional
from datetime import datetime

from .models.farm_context import FarmContext
from .models.agent_response import AgentResponse, EmergencyResponse, BatchResponse
from .agents.pest_agent import PestAgent
from .agents.weather_agent import WeatherAgent
from .agents.resource_agent import ResourceAgent
from .agents.market_agent import MarketAgent
from .services.groq_service import GroqService
from .services.notification_service import NotificationService
from .config import settings
from .exceptions import AgriGuardException, EmergencyTimeoutException


class AgriGuardSystem:
    """Main AgriGuard system orchestrator"""
    
    def __init__(self):
        self.groq_service = GroqService()
        self.notification_service = NotificationService()
        
        # Initialize specialized agents
        self.agents = {
            'pest': PestAgent(self.groq_service),
            'weather': WeatherAgent(self.groq_service),
            'resource': ResourceAgent(self.groq_service),
            'market': MarketAgent(self.groq_service)
        }
        
        # Performance tracking
        self.response_times = []
        self.emergency_count = 0
        
    async def process_query(self, query: str, context: FarmContext, 
                          agent_type: Optional[str] = None) -> AgentResponse:
        """Process a single query with real-time response"""
        
        start_time = time.time()
        
        try:
            # Determine urgency and select appropriate agent
            if agent_type:
                agent = self.agents.get(agent_type)
                if not agent:
                    raise AgriGuardException(f"Unknown agent type: {agent_type}")
            else:
                agent = await self._select_best_agent(query, context)
            
            # Process with selected agent
            response = await agent.process_query(query, context)
            
            # Track performance
            response_time = (time.time() - start_time) * 1000  # ms
            response.response_time_ms = response_time
            self.response_times.append(response_time)
            
            # Handle emergency responses
            if response.is_emergency:
                self.emergency_count += 1
                await self._handle_emergency_response(response, context)
            
            # Check response time limits for emergencies
            if context.is_emergency() and response_time > settings.max_response_time * 1000:
                raise EmergencyTimeoutException(response_time/1000, settings.max_response_time)
            
            return response
            
        except Exception as e:
            # Create error response
            error_response = AgentResponse(
                agent_name="system",
                response_type="emergency" if context.is_emergency() else "information",
                summary=f"System error: {str(e)}",
                detailed_response=f"An error occurred while processing your query: {str(e)}",
                confidence_score="low",
                response_time_ms=(time.time() - start_time) * 1000
            )
            return error_response
    
    async def process_batch_queries(self, queries: List[Dict[str, str]], 
                                  context: FarmContext) -> BatchResponse:
        """Process multiple queries simultaneously"""
        
        start_time = time.time()
        tasks = []
        
        # Create async tasks for all queries
        for query_info in queries:
            task = asyncio.create_task(
                self.process_query(
                    query_info['query'], 
                    context, 
                    query_info.get('agent_type')
                )
            )
            tasks.append(task)
        
        # Execute all queries concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful responses
        valid_responses = [r for r in responses if isinstance(r, AgentResponse)]
        
        # Create batch response
        batch = BatchResponse(
            query=f"Batch of {len(queries)} queries",
            responses=valid_responses,
            total_response_time_ms=(time.time() - start_time) * 1000
        )
        
        # Set primary response (highest urgency)
        if valid_responses:
            primary = max(valid_responses, key=lambda x: x.calculate_urgency_score())
            batch.primary_response = primary.response_id
        
        return batch
    
    async def _select_best_agent(self, query: str, context: FarmContext) -> object:
        """Select the most appropriate agent based on query content"""
        
        query_lower = query.lower()
        
        # Emergency pest detection keywords
        pest_keywords = ['pest', 'insect', 'bug', 'caterpillar', 'aphid', 'worm', 
                        'disease', 'fungus', 'leaf', 'damage', 'eating']
        
        # Weather emergency keywords  
        weather_keywords = ['weather', 'rain', 'storm', 'hail', 'wind', 'temperature',
                           'frost', 'drought', 'flood', 'climate']
        
        # Resource optimization keywords
        resource_keywords = ['water', 'irrigation', 'fertilizer', 'nutrient', 'soil',
                           'efficiency', 'cost', 'resource', 'optimize']
        
        # Market intelligence keywords
        market_keywords = ['price', 'sell', 'market', 'buyer', 'profit', 'income',
                          'harvest', 'trade', 'export']
        
        # Score each agent type
        scores = {
            'pest': sum(1 for keyword in pest_keywords if keyword in query_lower),
            'weather': sum(1 for keyword in weather_keywords if keyword in query_lower),
            'resource': sum(1 for keyword in resource_keywords if keyword in query_lower),
            'market': sum(1 for keyword in market_keywords if keyword in query_lower)
        }
        
        # Select highest scoring agent (default to pest for emergencies)
        best_agent_type = max(scores.keys(), key=lambda k: scores[k])
        
        # Override for high emergency situations
        if context.emergency_level >= 4 and scores['pest'] > 0:
            best_agent_type = 'pest'
        
        return self.agents[best_agent_type]
    
    async def _handle_emergency_response(self, response: AgentResponse, context: FarmContext):
        """Handle emergency responses with notifications"""
        
        if response.escalation_required:
            # Send immediate notifications
            await self.notification_service.send_emergency_alert(
                farmer_id=context.farmer.farmer_id,
                alert_data={
                    'title': f'URGENT: {response.summary}',
                    'message': response.detailed_response[:200],
                    'actions': response.immediate_actions,
                    'urgency': response.calculate_urgency_score()
                }
            )
        
        # Log emergency for analysis
        emergency_log = {
            'timestamp': datetime.now().isoformat(),
            'farmer_id': context.farmer.farmer_id,
            'emergency_level': response.emergency_level,
            'agent': response.agent_name,
            'response_time_ms': response.response_time_ms,
            'summary': response.summary
        }
    
    def get_performance_stats(self) -> Dict:
        """Get system performance statistics"""
        if not self.response_times:
            return {"status": "No queries processed yet"}
        
        avg_response_time = sum(self.response_times) / len(self.response_times)
        
        return {
            "total_queries": len(self.response_times),
            "emergency_count": self.emergency_count,
            "avg_response_time_ms": round(avg_response_time, 2),
            "fastest_response_ms": min(self.response_times),
            "slowest_response_ms": max(self.response_times),
            "sub_second_responses": len([t for t in self.response_times if t < 1000]),
            "emergency_response_rate": f"{(self.emergency_count/len(self.response_times)*100):.1f}%" if self.response_times else "0%"
        }


# Demo function for hackathon
async def demo_real_time_agriculture():
    """Demo function showcasing real-time agriculture assistance"""
    
    # Initialize system
    system = AgriGuardSystem()
    
    # Sample farmer context
    from .models.farm_context import FarmerProfile, Location, CropInfo, CropType, Season
    
    farmer = FarmerProfile(
        farmer_id="demo_farmer_001",
        name="Rajesh Kumar",
        phone="+91-9876543210",
        preferred_language="hi"
    )
    
    location = Location(
        latitude=30.7333,
        longitude=76.7794,
        address="Village Kharar, District Mohali",
        state="Punjab",
        country="India"
    )
    
    crop = CropInfo(
        crop_type=CropType.WHEAT,
        variety="PBW 343",
        planting_date=datetime(2024, 11, 15),
        growth_stage="flowering",
        area_planted=5.0
    )
    
    context = FarmContext(
        farmer=farmer,
        location=location,
        farm_size_total=5.0,
        current_crops=[crop],
        current_season=Season.RABI,
        emergency_level=4  # High emergency
    )
    
    # Demo scenarios
    scenarios = [
        {
            "query": "I see small green insects all over my wheat plants, they're multiplying very fast!",
            "agent_type": "pest",
            "expected_response_time": "< 1 second"
        },
        {
            "query": "Weather forecast shows hail storm tonight, what should I do immediately?",
            "agent_type": "weather", 
            "expected_response_time": "< 1 second"
        },
        {
            "query": "My irrigation system is using too much water and electricity costs are high",
            "agent_type": "resource",
            "expected_response_time": "< 5 seconds"
        },
        {
            "query": "Should I harvest my wheat now or wait? Current market price is ₹2,200 per quintal",
            "agent_type": "market",
            "expected_response_time": "< 5 seconds"
        }
    ]
    
    print("🌾 REAL-TIME AGRIGUARD DEMO 🌾")
    print("=" * 50)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 SCENARIO {i}: {scenario['agent_type'].upper()} EMERGENCY")
        print(f"Query: {scenario['query']}")
        print(f"Expected: {scenario['expected_response_time']}")
        print("-" * 30)
        
        start_time = time.time()
        response = await system.process_query(
            scenario['query'], 
            context, 
            scenario['agent_type']
        )
        end_time = time.time()
        
        print(f"⚡ Response Time: {(end_time - start_time)*1000:.0f}ms")
        print(f"🎯 Agent: {response.agent_name}")
        print(f"📊 Urgency Score: {response.calculate_urgency_score()}/10")
        print(f"💡 Summary: {response.summary}")
        
        if response.immediate_actions:
            print("🚨 IMMEDIATE ACTIONS:")
            for action in response.immediate_actions[:3]:
                print(f"   • {action}")
        
        if response.is_emergency:
            print("🔴 EMERGENCY ALERT TRIGGERED")
    
    # Performance summary
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("=" * 30)
    stats = system.get_performance_stats()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    # Set environment variables for demo
    os.environ.setdefault("GROQ_API_KEY", "demo_key_replace_with_real")
    
    # Run demo
    asyncio.run(demo_real_time_agriculture())
