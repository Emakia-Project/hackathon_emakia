import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Any
import hashlib
import time
from dataclasses import dataclass
from collections import defaultdict
import redis
import openai
from openai import AsyncOpenAI
import re
import os
import uuid
from dotenv import load_dotenv
import praw
import google.generativeai as genai

# Google ADK imports
from google.adk.sessions import InMemorySessionService
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.genai import types
from google.oauth2 import service_account
from google.cloud import bigquery

# Local imports
from inputs.reddit_scraper import get_reddit_posts
from inputs.maxnews_scraper import get_maxnews_articles
from inputs.bigquery_loader import get_tweets_from_bigquery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key"))

# Configure Gemini API
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    st.warning("Missing GOOGLE_API_KEY in environment.")

@dataclass
class UnifiedModerationResult:
    """Unified data class for moderation results from both systems"""
    content_id: str
    platform: str
    content: str
    toxicity_score: float
    bias_score: float
    misinformation_score: float
    sentiment_score: float
    timestamp: datetime
    metadata: Dict
    flagged: bool
    confidence: float
    llm_reasoning: str
    categories: List[str]
    analysis_method: str  # "openai", "gemini", or "hybrid"
    raw_analysis: Dict  # Store raw analysis from both systems

class UnifiedContentModerationSystem:
    """Unified content moderation system combining OpenAI and Google Gemini approaches"""
    
    def __init__(self):
        self.redis_client = None
        self.moderation_history = []
        self.alert_threshold = 0.7
        self.batch_size = 50
        self.openai_client = openai_client
        
        # Initialize Google ADK agents
        self.gemini_agents = self._initialize_gemini_agents()
        
        # LLM prompts for OpenAI analysis
        self._initialize_openai_prompts()
        
    def _initialize_gemini_agents(self):
        """Initialize Google ADK agents"""
        try:
            toxicity_agent = LlmAgent(
                name="ToxicityAnalyst",
                model="gemini-2.0-flash",
                instruction="""You are an AI Toxicity Analyst. Analyze the following statement and determine if it contains toxic language.
                Respond with either 'toxic' or 'non-toxic' and provide a confidence score from 0.0 to 1.0.""",
                description="Analyzes statements for toxicity.",
                output_key="toxicity_analysis"
            )
            
            bias_agent = LlmAgent(
                name="BiasAnalyst",
                model="gemini-2.0-flash",
                instruction="""You are an AI Bias Analyst. Assess the following statement for bias.
                Indicate whether the statement contains 'bias' or is 'neutral' and provide a confidence score from 0.0 to 1.0.""",
                description="Analyzes statements for bias.",
                output_key="bias_analysis"
            )
            
            misinfo_agent = LlmAgent(
                name="MisinformationAnalyst",
                model="gemini-2.0-flash",
                instruction="""You are a fake news analyst. Determine if the following fact is fake news.
                Respond with either 'misinformation' or 'accurate' and provide a confidence score from 0.0 to 1.0.""",
                description="Detects misinformation in statements.",
                output_key="misinformation_analysis"
            )
            
            return ParallelAgent(
                name="GeminiAnalysisAgent",
                sub_agents=[toxicity_agent, bias_agent, misinfo_agent]
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini agents: {e}")
            return None
    
    def _initialize_openai_prompts(self):
        """Initialize OpenAI analysis prompts"""
        self.toxicity_prompt = """
        Analyze the following content for toxicity, hate speech, harassment, and harmful language.
        
        Content: "{content}"
        Platform: {platform}
        
        Provide a score from 0.0 to 1.0 (where 1.0 is extremely toxic) and explain your reasoning.
        Consider:
        - Hate speech and discriminatory language
        - Personal attacks and harassment
        - Threats or violent content
        - Profanity and offensive language
        - Context and cultural sensitivity
        
        Return your response in JSON format:
        {{
            "toxicity_score": 0.0,
            "reasoning": "explanation",
            "categories": ["category1", "category2"],
            "confidence": 0.0
        }}
        """
        
        self.bias_prompt = """
        Analyze the following content for bias, unfairness, and discriminatory perspectives.
        
        Content: "{content}"
        Platform: {platform}
        
        Evaluate for various types of bias including:
        - Gender bias
        - Racial/ethnic bias
        - Religious bias
        - Political bias
        - Socioeconomic bias
        - Age bias
        - Cultural bias
        
        Provide a score from 0.0 to 1.0 (where 1.0 is extremely biased) and explain your reasoning.
        
        Return your response in JSON format:
        {{
            "bias_score": 0.0,
            "bias_types": ["type1", "type2"],
            "reasoning": "explanation",
            "confidence": 0.0
        }}
        """
        
        self.misinformation_prompt = """
        Analyze the following content for potential misinformation, fake news, and misleading claims.
        
        Content: "{content}"
        Platform: {platform}
        
        Evaluate for:
        - Factual accuracy concerns
        - Misleading headlines or clickbait
        - Conspiracy theories
        - Unverified claims presented as facts
        - Manipulation of data or statistics
        - Lack of credible sources
        - Emotional manipulation techniques
        
        Provide a score from 0.0 to 1.0 (where 1.0 is definitely misinformation) and explain your reasoning.
        
        Return your response in JSON format:
        {{
            "misinformation_score": 0.0,
            "risk_factors": ["factor1", "factor2"],
            "reasoning": "explanation",
            "confidence": 0.0,
            "fact_check_suggestions": ["suggestion1", "suggestion2"]
        }}
        """
        
        self.sentiment_prompt = """
        Analyze the sentiment and emotional tone of the following content.
        
        Content: "{content}"
        Platform: {platform}
        
        Provide sentiment analysis considering:
        - Overall emotional tone
        - Intensity of emotions
        - Context and cultural factors
        - Sarcasm or irony detection
        
        Return sentiment score from -1.0 (very negative) to 1.0 (very positive).
        
        Return your response in JSON format:
        {{
            "sentiment_score": 0.0,
            "emotions": ["emotion1", "emotion2"],
            "intensity": 0.0,
            "reasoning": "explanation"
        }}
        """
    
    def initialize_redis(self, redis_host='localhost', redis_port=6379):
        """Initialize Redis connection for caching"""
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def analyze_with_openai(self, content: str, platform: str) -> Dict:
        """Analyze content using OpenAI GPT models"""
        try:
            # Run all analyses concurrently
            tasks = [
                self._openai_analyze(self.toxicity_prompt, content, platform, "toxicity"),
                self._openai_analyze(self.bias_prompt, content, platform, "bias"),
                self._openai_analyze(self.misinformation_prompt, content, platform, "misinformation"),
                self._openai_analyze(self.sentiment_prompt, content, platform, "sentiment")
            ]
            
            results = await asyncio.gather(*tasks)
            
            return {
                "toxicity": results[0],
                "bias": results[1],
                "misinformation": results[2],
                "sentiment": results[3],
                "method": "openai"
            }
        except Exception as e:
            logger.error(f"OpenAI analysis error: {e}")
            return self._get_fallback_analysis("openai")
    
    async def analyze_with_gemini(self, content: str, platform: str) -> Dict:
        """Analyze content using Google Gemini ADK agents"""
        if not self.gemini_agents:
            return self._get_fallback_analysis("gemini")
        
        try:
            session_service = InMemorySessionService()
            session = session_service.create_session(
                app_name="unified_moderation",
                user_id=f"user_{uuid.uuid4()}",
                session_id=f"session_{uuid.uuid4()}"
            )
            
            runner = Runner(agent=self.gemini_agents, app_name="unified_moderation", session_service=session_service)
            content_obj = types.Content(role="user", parts=[types.Part(text=content)])
            events = runner.run(user_id=session.user_id, session_id=session.id, new_message=content_obj)
            
            # Parse Gemini results
            results = {"toxicity": {}, "bias": {}, "misinformation": {}, "sentiment": {}}
            
            for event in events:
                if event and event.content.parts:
                    output = event.content.parts[0].text.lower()
                    if "toxic" in output:
                        results["toxicity"] = self._parse_gemini_output(output, "toxicity")
                    elif "bias" in output:
                        results["bias"] = self._parse_gemini_output(output, "bias")
                    elif "misinformation" in output or "accurate" in output:
                        results["misinformation"] = self._parse_gemini_output(output, "misinformation")
            
            # Add sentiment analysis (simple fallback)
            results["sentiment"] = self._simple_sentiment_analysis(content)
            results["method"] = "gemini"
            
            return results
            
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return self._get_fallback_analysis("gemini")
    
    def _parse_gemini_output(self, output: str, analysis_type: str) -> Dict:
        """Parse Gemini agent output into structured format"""
        # Simple parsing - in production, you'd want more sophisticated parsing
        is_positive = any(word in output for word in ["toxic", "bias", "misinformation"])
        confidence = 0.8 if is_positive else 0.2
        
        return {
            f"{analysis_type}_score": 1.0 if is_positive else 0.0,
            "confidence": confidence,
            "reasoning": output,
            "categories": [analysis_type] if is_positive else []
        }
    
    def _simple_sentiment_analysis(self, content: str) -> Dict:
        """Simple sentiment analysis fallback"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'happy', 'joy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'disappointed', 'horrible']
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / max(len(words), 1)
        
        return {
            "sentiment_score": max(-1.0, min(1.0, sentiment_score)),
            "confidence": 0.5,
            "reasoning": "Simple word-based sentiment analysis",
            "emotions": ["positive"] if sentiment_score > 0 else ["negative"] if sentiment_score < 0 else ["neutral"]
        }
    
    def _get_fallback_analysis(self, method: str) -> Dict:
        """Get fallback analysis when API calls fail"""
        return {
            "toxicity": {"toxicity_score": 0.0, "confidence": 0.0, "reasoning": "Analysis failed", "categories": []},
            "bias": {"bias_score": 0.0, "confidence": 0.0, "reasoning": "Analysis failed", "bias_types": []},
            "misinformation": {"misinformation_score": 0.0, "confidence": 0.0, "reasoning": "Analysis failed", "risk_factors": []},
            "sentiment": {"sentiment_score": 0.0, "confidence": 0.0, "reasoning": "Analysis failed", "emotions": []},
            "method": method
        }
    
    async def _openai_analyze(self, prompt: str, content: str, platform: str, analysis_type: str) -> Dict:
        """Generic OpenAI analysis function"""
        try:
            formatted_prompt = prompt.format(content=content, platform=platform)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert content moderator and analyst. Provide accurate, unbiased analysis in the requested JSON format."},
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"OpenAI {analysis_type} analysis error: {e}")
            return {
                f"{analysis_type}_score": 0.0,
                "reasoning": f"Analysis failed: {str(e)}",
                "confidence": 0.0,
                "categories": ["error"]
            }
    
    async def unified_analysis(self, content: str, platform: str, use_both: bool = True) -> Dict:
        """Perform unified analysis using both OpenAI and Gemini"""
        if use_both:
            # Run both analyses concurrently
            openai_task = self.analyze_with_openai(content, platform)
            gemini_task = self.analyze_with_gemini(content, platform)
            
            openai_results, gemini_results = await asyncio.gather(
                openai_task, gemini_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(openai_results, Exception):
                openai_results = self._get_fallback_analysis("openai")
            if isinstance(gemini_results, Exception):
                gemini_results = self._get_fallback_analysis("gemini")
            
            # Combine results (weighted average)
            combined_results = self._combine_analysis_results(openai_results, gemini_results)
            combined_results["method"] = "hybrid"
            return combined_results
        else:
            # Use OpenAI by default
            return await self.analyze_with_openai(content, platform)
    
    def _combine_analysis_results(self, openai_results: Dict, gemini_results: Dict) -> Dict:
        """Combine results from both analysis methods"""
        combined = {}
        
        for analysis_type in ["toxicity", "bias", "misinformation", "sentiment"]:
            openai_data = openai_results.get(analysis_type, {})
            gemini_data = gemini_results.get(analysis_type, {})
            
            # Weighted combination (OpenAI gets 70% weight, Gemini 30%)
            openai_weight = 0.7
            gemini_weight = 0.3
            
            score_key = f"{analysis_type}_score"
            openai_score = openai_data.get(score_key, 0.0)
            gemini_score = gemini_data.get(score_key, 0.0)
            
            combined_score = (openai_score * openai_weight + gemini_score * gemini_weight)
            
            # Combine confidence scores
            openai_conf = openai_data.get("confidence", 0.0)
            gemini_conf = gemini_data.get("confidence", 0.0)
            combined_conf = (openai_conf * openai_weight + gemini_conf * gemini_weight)
            
            # Combine reasoning
            combined_reasoning = f"OpenAI: {openai_data.get('reasoning', 'N/A')}\nGemini: {gemini_data.get('reasoning', 'N/A')}"
            
            # Combine categories
            combined_categories = list(set(
                openai_data.get("categories", []) + 
                openai_data.get("bias_types", []) + 
                openai_data.get("risk_factors", []) + 
                openai_data.get("emotions", []) +
                gemini_data.get("categories", [])
            ))
            
            combined[analysis_type] = {
                score_key: combined_score,
                "confidence": combined_conf,
                "reasoning": combined_reasoning,
                "categories": combined_categories
            }
        
        return combined
    
    def generate_content_id(self, content: str, platform: str) -> str:
        """Generate unique content ID"""
        return hashlib.md5(f"{content}_{platform}_{datetime.now()}".encode()).hexdigest()[:12]
    
    async def moderate_content(self, content: str, platform: str, metadata: Dict = None, use_both: bool = True) -> UnifiedModerationResult:
        """Comprehensive unified content moderation"""
        if metadata is None:
            metadata = {}
            
        content_id = self.generate_content_id(content, platform)
        
        # Check cache first
        cached_result = None
        if self.redis_client:
            cached_key = f"unified_moderation:{hashlib.md5(content.encode()).hexdigest()}"
            cached_result = self.redis_client.get(cached_key)
            if cached_result:
                cached_data = json.loads(cached_result)
                return UnifiedModerationResult(**cached_data)
        
        # Perform unified analysis
        start_time = time.time()
        analysis_results = await self.unified_analysis(content, platform, use_both)
        processing_time = time.time() - start_time
        
        # Extract scores
        toxicity_score = analysis_results["toxicity"].get("toxicity_score", 0.0)
        bias_score = analysis_results["bias"].get("bias_score", 0.0)
        misinformation_score = analysis_results["misinformation"].get("misinformation_score", 0.0)
        sentiment_score = analysis_results["sentiment"].get("sentiment_score", 0.0)
        
        # Calculate overall confidence and flagging
        avg_confidence = np.mean([
            analysis_results["toxicity"].get("confidence", 0.0),
            analysis_results["bias"].get("confidence", 0.0),
            analysis_results["misinformation"].get("confidence", 0.0)
        ])
        
        # Flag if any score exceeds threshold
        flagged = any([
            toxicity_score > self.alert_threshold,
            bias_score > self.alert_threshold,
            misinformation_score > self.alert_threshold
        ])
        
        # Collect all categories
        categories = []
        for analysis_type in ["toxicity", "bias", "misinformation", "sentiment"]:
            categories.extend(analysis_results[analysis_type].get("categories", []))
        
        # Combine reasoning
        combined_reasoning = {
            "processing_time": f"{processing_time:.2f}s",
            "analysis_method": analysis_results.get("method", "unknown"),
            "detailed_analysis": analysis_results
        }
        
        result = UnifiedModerationResult(
            content_id=content_id,
            platform=platform,
            content=content[:500] + "..." if len(content) > 500 else content,
            toxicity_score=toxicity_score,
            bias_score=bias_score,
            misinformation_score=misinformation_score,
            sentiment_score=sentiment_score,
            timestamp=datetime.now(),
            metadata=metadata,
            flagged=flagged,
            confidence=avg_confidence,
            llm_reasoning=json.dumps(combined_reasoning, indent=2),
            categories=list(set(categories)),
            analysis_method=analysis_results.get("method", "unknown"),
            raw_analysis=analysis_results
        )
        
        # Cache result
        if self.redis_client and not cached_result:
            self.redis_client.setex(
                cached_key, 
                7200,  # 2 hour cache
                json.dumps(result.__dict__, default=str)
            )
        
        self.moderation_history.append(result)
        return result
    
    async def batch_moderate(self, contents: List[Dict], use_both: bool = True) -> List[UnifiedModerationResult]:
        """Batch moderation with unified analysis"""
        results = []
        
        # Process in smaller batches
        for i in range(0, len(contents), self.batch_size):
            batch = contents[i:i + self.batch_size]
            
            # Add delay between batches
            if i > 0:
                await asyncio.sleep(1)
            
            batch_tasks = []
            for item in batch:
                task = self.moderate_content(
                    content=item['content'],
                    platform=item['platform'],
                    metadata=item.get('metadata', {}),
                    use_both=use_both
                )
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Filter successful results
            for result in batch_results:
                if not isinstance(result, Exception):
                    results.append(result)
                else:
                    logger.error(f"Batch processing error: {result}")
        
        return results
    
    def get_moderation_stats(self) -> Dict:
        """Get comprehensive moderation statistics"""
        if not self.moderation_history:
            return {}
        
        df = pd.DataFrame([
            {
                'platform': r.platform,
                'toxicity_score': r.toxicity_score,
                'bias_score': r.bias_score,
                'misinformation_score': r.misinformation_score,
                'sentiment_score': r.sentiment_score,
                'flagged': r.flagged,
                'timestamp': r.timestamp,
                'confidence': r.confidence,
                'analysis_method': r.analysis_method,
                'categories': len(r.categories)
            }
            for r in self.moderation_history
        ])
        
        stats = {
            'total_analyzed': len(df),
            'flagged_content': len(df[df['flagged'] == True]),
            'avg_toxicity': df['toxicity_score'].mean(),
            'avg_bias': df['bias_score'].mean(),
            'avg_misinformation': df['misinformation_score'].mean(),
            'avg_sentiment': df['sentiment_score'].mean(),
            'platform_breakdown': df['platform'].value_counts().to_dict(),
            'method_breakdown': df['analysis_method'].value_counts().to_dict(),
            'hourly_analysis': df.groupby(df['timestamp'].dt.hour).size().to_dict(),
            'high_risk_content': len(df[df['confidence'] > 0.8]),
            'category_distribution': {}
        }
        
        # Category distribution
        all_categories = []
        for r in self.moderation_history:
            all_categories.extend(r.categories)
        
        from collections import Counter
        category_counts = Counter(all_categories)
        stats['category_distribution'] = dict(category_counts.most_common(10))
        
        return stats

# Enhanced Streamlit UI for Unified System
class UnifiedModerationDashboard:
    """Unified dashboard supporting both OpenAI and Gemini analysis"""
    
    def __init__(self, moderation_system: UnifiedContentModerationSystem):
        self.system = moderation_system
        
    def render_main_dashboard(self):
        """Render main dashboard"""
        st.set_page_config(
            page_title="Unified AI Content Moderation System",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 Unified AI Content Moderation System")
        st.markdown("### Advanced multi-LLM analysis combining OpenAI GPT and Google Gemini")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("🔧 Configuration")
            
            # API Key inputs
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.get("openai_api_key", ""),
                help="Enter your OpenAI API key"
            )
            
            if openai_key:
                st.session_state["openai_api_key"] = openai_key
                os.environ["OPENAI_API_KEY"] = openai_key
                self.system.openai_client.api_key = openai_key
                st.success("OpenAI API key configured!")
            
            google_key = st.text_input(
                "Google API Key",
                type="password",
                value=st.session_state.get("google_api_key", ""),
                help="Enter your Google API key for Gemini"
            )
            
            if google_key:
                st.session_state["google_api_key"] = google_key
                os.environ["GOOGLE_API_KEY"] = google_key
                st.success("Google API key configured!")
            
            # Analysis method selection
            analysis_method = st.selectbox(
                "Analysis Method",
                ["Hybrid (Both)", "OpenAI Only", "Gemini Only"],
                index=0,
                help="Choose which LLM to use for analysis"
            )
            
            # Alert threshold
            threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.7, 0.1)
            self.system.alert_threshold = threshold
            
            # Platform filter
            platforms = ['All', 'Reddit', 'Twitter', 'News', 'Custom']
            selected_platform = st.selectbox("Platform Filter", platforms)
            
            # Processing options
            st.subheader("Processing Options")
            batch_size = st.slider("Batch Size", 1, 20, 5)
            self.system.batch_size = batch_size
            
            # Redis connection
            if st.button("Connect to Redis"):
                self.system.initialize_redis()
                st.success("Redis connection attempted")
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🔍 Single Analysis", 
            "📊 Batch Processing", 
            "📈 Analytics", 
            "🚨 Alerts", 
            "🧠 LLM Insights",
            "📰 Data Sources",
            "⚙️ System Status"
        ])
        
        with tab1:
            self.render_single_analysis(analysis_method)
            
        with tab2:
            self.render_batch_processing(analysis_method)
            
        with tab3:
            self.render_analytics()
            
        with tab4:
            self.render_alerts()
            
        with tab5:
            self.render_llm_insights()
            
        with tab6:
            self.render_data_sources()
            
        with tab7:
            self.render_system_status()
    
    def render_single_analysis(self, analysis_method: str):
        """Render single content analysis interface"""
        st.header("🔍 AI Content Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            content = st.text_area(
                "Enter content to analyze:",
                height=150,
                placeholder="Paste text from Reddit, Twitter, news articles, etc. The AI will analyze for toxicity, bias, misinformation, and sentiment."
            )
            
            platform = st.selectbox("Platform", ['Reddit', 'Twitter', 'News', 'Custom'])
            
            metadata = {}
            with st.expander("📋 Additional Metadata"):
                author = st.text_input("Author/Username")
                url = st.text_input("Source URL")
                context = st.text_area("Additional Context", height=100)
                if author:
                    metadata['author'] = author
                if url:
                    metadata['url'] = url
                if context:
                    metadata['context'] = context
        
        with col2:
            st.subheader("Analysis Options")
            
            analysis_types = st.multiselect(
                "Analysis Types",
                ["Toxicity", "Bias", "Misinformation", "Sentiment"],
                default=["Toxicity", "Bias", "Misinformation", "Sentiment"]
            )
            
            detailed_reasoning = st.checkbox("Show Detailed LLM Reasoning", value=True)
            
            if st.button("🤖 Analyze with AI", type="primary", use_container_width=True):
                if content.strip():
                    # Check API keys based on method
                    use_both = analysis_method == "Hybrid (Both)"
                    use_openai = analysis_method in ["Hybrid (Both)", "OpenAI Only"]
                    use_gemini = analysis_method in ["Hybrid (Both)", "Gemini Only"]
                    
                    if use_openai and not os.getenv("OPENAI_API_KEY"):
                        st.error("Please configure your OpenAI API key in the sidebar!")
                        return
                    
                    if use_gemini and not os.getenv("GOOGLE_API_KEY"):
                        st.error("Please configure your Google API key in the sidebar!")
                        return
                    
                    with st.spinner("AI is analyzing content... This may take a few moments."):
                        try:
                            result = asyncio.run(self.system.moderate_content(content, platform, metadata, use_both))
                            self.display_unified_moderation_result(result, detailed_reasoning)
                            
                            # Check for alerts
                            if result.flagged:
                                st.error(f"🚨 Content Flagged for Review (Confidence: {result.confidence:.2f})")
                        
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                else:
                    st.warning("Please enter content to analyze")
    
    def render_batch_processing(self, analysis_method: str):
        """Render batch processing interface"""
        st.header("📊 Batch AI Processing")
        
        st.warning("⚠️ Batch processing uses multiple LLM calls and may incur significant API costs. Monitor your usage!")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with content",
            type=['csv'],
            help="CSV should have columns: 'content', 'platform', 'metadata' (optional)"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} rows")
            st.dataframe(df.head())
            
            if len(df) > 100:
                st.error("Batch size limited to 100 items for API cost control")
                return
            
            if st.button("🤖 Process Batch with AI", type="primary"):
                contents = []
                for _, row in df.iterrows():
                    item = {
                        'content': row['content'],
                        'platform': row.get('platform', 'Unknown'),
                        'metadata': json.loads(row.get('metadata', '{}')) if 'metadata' in row else {}
                    }
                    contents.append(item)
                
                use_both = analysis_method == "Hybrid (Both)"
                
                with st.spinner(f"AI processing {len(contents)} items... This will take several minutes."):
                    try:
                        results = asyncio.run(self.system.batch_moderate(contents, use_both))
                        
                        # Create results dataframe
                        results_data = []
                        for result in results:
                            results_data.append({
                                'Content ID': result.content_id,
                                'Platform': result.platform,
                                'Toxicity': f"{result.toxicity_score:.3f}",
                                'Bias': f"{result.bias_score:.3f}",
                                'Misinformation': f"{result.misinformation_score:.3f}",
                                'Sentiment': f"{result.sentiment_score:.3f}",
                                'Flagged': "🚨" if result.flagged else "✅",
                                'Confidence': f"{result.confidence:.3f}",
                                'Method': result.analysis_method,
                                'Categories': ", ".join(result.categories[:3])
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.success(f"AI processed {len(results)} items")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "unified_moderation_results.csv",
                            "text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Batch processing failed: {str(e)}")
    
    def render_analytics(self):
        """Render analytics dashboard"""
        st.header("📈 Advanced Analytics")
        
        stats = self.system.get_moderation_stats()
        
        if not stats:
            st.info("No data available. Analyze some content first!")
            return
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Analyzed", stats['total_analyzed'])
            
        with col2:
            st.metric("Flagged Content", stats['flagged_content'])
            
        with col3:
            st.metric("Avg Toxicity", f"{stats['avg_toxicity']:.3f}")
            
        with col4:
            st.metric("Avg Bias", f"{stats['avg_bias']:.3f}")
            
        with col5:
            st.metric("Avg Misinformation", f"{stats['avg_misinformation']:.3f}")
        
        # Method breakdown
        if stats.get('method_breakdown'):
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=list(stats['method_breakdown'].values()),
                    names=list(stats['method_breakdown'].keys()),
                    title="Analysis Method Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if stats.get('category_distribution'):
                    fig = px.bar(
                        x=list(stats['category_distribution'].values()),
                        y=list(stats['category_distribution'].keys()),
                        orientation='h',
                        title="Top Content Categories Detected"
                    )
                    fig.update_yaxis(title="Categories")
                    fig.update_xaxis(title="Frequency")
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts(self):
        """Render alerts dashboard"""
        st.header("🚨 Alert Management")
        
        # Generate alerts for flagged content
        alerts = []
        for result in self.system.moderation_history:
            if result.flagged and result.confidence > 0.7:
                alerts.append({
                    'level': 'HIGH' if result.confidence > 0.8 else 'MEDIUM',
                    'content_id': result.content_id,
                    'platform': result.platform,
                    'confidence': result.confidence,
                    'timestamp': result.timestamp,
                    'analysis_method': result.analysis_method,
                    'categories': result.categories
                })
        
        if not alerts:
            st.success("No active alerts!")
            return
        
        st.warning(f"Found {len(alerts)} active alerts")
        
        # Display alerts
        for i, alert in enumerate(alerts):
            with st.expander(f"{alert['level']} Alert - {alert['content_id']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Platform:** {alert['platform']}")
                    st.write(f"**Confidence:** {alert['confidence']:.3f}")
                    st.write(f"**Method:** {alert['analysis_method']}")
                    st.write(f"**Time:** {alert['timestamp']}")
                    st.write(f"**Categories:** {', '.join(alert['categories'])}")
                
                with col2:
                    if st.button(f"Resolve Alert {i+1}"):
                        st.success("Alert marked as resolved")
    
    def render_llm_insights(self):
        """Render LLM insights and reasoning"""
        st.header("🧠 LLM Analysis Insights")
        
        if not self.system.moderation_history:
            st.info("No analysis data available. Analyze some content first!")
            return
        
        # Show recent analyses
        recent_results = self.system.moderation_history[-5:]  # Last 5 analyses
        
        for i, result in enumerate(recent_results):
            with st.expander(f"Analysis {i+1} - {result.content_id} ({result.analysis_method})"):
                st.write(f"**Content:** {result.content}")
                st.write(f"**Platform:** {result.platform}")
                st.write(f"**Analysis Method:** {result.analysis_method}")
                
                # Display scores
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Toxicity", f"{result.toxicity_score:.3f}")
                with col2:
                    st.metric("Bias", f"{result.bias_score:.3f}")
                with col3:
                    st.metric("Misinformation", f"{result.misinformation_score:.3f}")
                with col4:
                    st.metric("Sentiment", f"{result.sentiment_score:.3f}")
                
                # Show detailed reasoning
                if st.checkbox(f"Show detailed reasoning for analysis {i+1}"):
                    try:
                        reasoning = json.loads(result.llm_reasoning)
                        st.json(reasoning)
                    except:
                        st.text(result.llm_reasoning)
    
    def render_data_sources(self):
        """Render data sources integration"""
        st.header("📰 Data Sources Integration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Reddit Posts")
            if st.button("Fetch Reddit Posts"):
                with st.spinner("Fetching Reddit posts..."):
                    posts = get_reddit_posts("politics", limit=3)
                    if posts:
                        st.success(f"Fetched {len(posts)} Reddit posts")
                        for post in posts:
                            st.write(f"**{post['title']}**")
                            st.write(f"Content: {post['content'][:200]}...")
                    else:
                        st.warning("No Reddit posts available")
        
        with col2:
            st.subheader("News Articles")
            if st.button("Fetch News Articles"):
                with st.spinner("Fetching news articles..."):
                    articles = get_maxnews_articles()
                    if articles:
                        st.success(f"Fetched {len(articles)} news articles")
                        for article in articles:
                            st.write(f"**{article['title']}**")
                            st.write(f"Content: {article['content'][:200]}...")
                    else:
                        st.warning("No news articles available")
        
        st.subheader("BigQuery Tweets")
        if st.button("Fetch Tweets from BigQuery"):
            with st.spinner("Fetching tweets from BigQuery..."):
                tweets = get_tweets_from_bigquery()
                if tweets:
                    st.success(f"Fetched {len(tweets)} tweets")
                    for tweet in tweets:
                        st.write(f"**Tweet ID: {tweet['title']}**")
                        st.write(f"Content: {tweet['content'][:200]}...")
                else:
                    st.warning("No tweets available")
    
    def render_system_status(self):
        """Render system status"""
        st.header("⚙️ System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("API Status")
            
            # Check API availability
            apis = {
                "OpenAI API": "🟢 Available" if os.getenv("OPENAI_API_KEY") else "🔴 Not Configured",
                "Google API": "🟢 Available" if os.getenv("GOOGLE_API_KEY") else "🔴 Not Configured",
                "Redis Cache": "🟢 Connected" if self.system.redis_client else "🔴 Disconnected"
            }
            
            for api, status in apis.items():
                st.write(f"**{api}:** {status}")
        
        with col2:
            st.subheader("System Metrics")
            
            stats = self.system.get_moderation_stats()
            if stats:
                st.metric("Total Requests", stats['total_analyzed'])
                st.metric("Flagged Content", stats['flagged_content'])
                st.metric("High Risk Items", stats.get('high_risk_content', 0))
            else:
                st.info("No metrics available yet")
    
    def display_unified_moderation_result(self, result: UnifiedModerationResult, detailed_reasoning: bool = True):
        """Display unified moderation result"""
        st.subheader(f"Analysis Results - {result.content_id}")
        
        # Status indicator
        if result.flagged:
            st.error("🚨 Content Flagged for Review")
        else:
            st.success("✅ Content Cleared")
        
        # Analysis method indicator
        method_colors = {
            "openai": "🔵",
            "gemini": "🟡", 
            "hybrid": "🟣"
        }
        st.write(f"**Analysis Method:** {method_colors.get(result.analysis_method, '⚪')} {result.analysis_method.upper()}")
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Toxicity", f"{result.toxicity_score:.3f}")
        
        with col2:
            st.metric("Bias", f"{result.bias_score:.3f}")
        
        with col3:
            st.metric("Misinformation", f"{result.misinformation_score:.3f}")
        
        with col4:
            sentiment_label = "Positive" if result.sentiment_score > 0 else "Negative" if result.sentiment_score < 0 else "Neutral"
            st.metric("Sentiment", f"{result.sentiment_score:.3f} ({sentiment_label})")
        
        with col5:
            st.metric("Confidence", f"{result.confidence:.3f}")
        
        # Categories
        if result.categories:
            st.write("**Detected Categories:**")
            for category in result.categories:
                st.write(f"- {category}")
        
        # Content preview
        with st.expander("Content Preview"):
            st.text(result.content)
        
        # Detailed reasoning
        if detailed_reasoning:
            with st.expander("Detailed LLM Reasoning"):
                try:
                    reasoning = json.loads(result.llm_reasoning)
                    st.json(reasoning)
                except:
                    st.text(result.llm_reasoning)
        
        # Metadata
        if result.metadata:
            with st.expander("Metadata"):
                st.json(result.metadata)

# Main application
def main():
    # Initialize unified system
    moderation_system = UnifiedContentModerationSystem()
    dashboard = UnifiedModerationDashboard(moderation_system)
    
    # Render dashboard
    dashboard.render_main_dashboard()

if __name__ == "__main__":
    main()
