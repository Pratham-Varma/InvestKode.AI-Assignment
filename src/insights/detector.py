"""
Real-Time Insight Detection Module

Implement your insight extraction logic here.

Requirements:
- Generate rolling summaries as transcript chunks arrive
- Detect key financial signals:
  - Revenue mentions
  - Guidance (forward-looking statements)
  - Risks and challenges
  - Outlook and projections
- Insights should update continuously, not just at the end

Suggested Approaches:
- LLM-based extraction (OpenAI, Anthropic, Google)
- Rule-based keyword extraction as fallback
- Hybrid: rules for speed, LLM for quality
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.transcription.transcriber import TranscriptChunk


class InsightType(Enum):
    """Types of financial insights to detect."""
    
    REVENUE = "revenue"
    GUIDANCE = "guidance"
    RISK = "risk"
    OUTLOOK = "outlook"
    GROWTH = "growth"
    MARGIN = "margin"
    MARKET_SHARE = "market_share"
    COMPETITION = "competition"
    REGULATION = "regulation"
    OTHER = "other"


class Sentiment(Enum):
    """Sentiment classification for insights."""
    
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class Insight:
    """Represents a detected insight."""
    
    type: InsightType
    text: str
    confidence: float = 1.0
    sentiment: Sentiment = Sentiment.NEUTRAL
    
    # Source information
    source_text: str = ""
    timestamp: float = 0.0


@dataclass
class InsightResult:
    """Result of insight detection for a transcript chunk."""
    
    # The transcript chunk that was analyzed
    chunk: TranscriptChunk
    
    # Detected insights
    insights: List[Insight] = field(default_factory=list)
    
    # Rolling summary (updated with each chunk)
    rolling_summary: str = ""
    
    # Key metrics mentioned
    metrics: dict = field(default_factory=dict)


class InsightDetector:
    """
    Real-time insight detection from transcript chunks.
    
    TODO: Implement this class with your chosen approach.
    
    Example usage:
        detector = InsightDetector()
        
        async for chunk in transcriber.process_audio(audio_path):
            result = await detector.analyze(chunk)
            print(f"Summary: {result.rolling_summary}")
            for insight in result.insights:
                print(f"  [{insight.type.value}] {insight.text}")
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the insight detector.
        
        Args:
            use_llm: Whether to use LLM for insight extraction
        """
        self.use_llm = use_llm
        self.conversation_history: List[TranscriptChunk] = []
        self.all_insights: List[Insight] = []
        self.current_summary: str = ""
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM client and sentiment analyzer
        self.client = None
        self.llm_provider = None
        self.sentiment_analyzer = None
        
        # Initialize sentiment analysis if enabled
        if os.getenv("USE_SENTIMENT_ANALYSIS", "true").lower() == "true":
            try:
                from transformers import pipeline
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if self._is_cuda_available() else -1
                )
                self.logger.info("Sentiment analyzer loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load sentiment analyzer: {e}")
        
        if self.use_llm:
            # Get LLM provider from environment
            llm_provider = os.getenv("LLM_PROVIDER", "rule-based").lower()
            
            if llm_provider == "local":
                self._initialize_local_llm()
            elif llm_provider == "gemini":
                self._initialize_gemini()
            elif llm_provider == "openai":
                self._initialize_openai()
            else:
                self.logger.info("Using rule-based extraction (LLM_PROVIDER=rule-based)")
                self.use_llm = False
            
            # If initialization failed, fall back to rule-based
            if not self.client and self.llm_provider != "local":
                self.logger.warning("LLM initialization failed. Falling back to rule-based extraction.")
                self.use_llm = False
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _initialize_local_llm(self):
        """Initialize local LLM with quantization."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            model_name = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
            quantization = os.getenv("LOCAL_LLM_QUANTIZATION", "4bit")
            
            self.logger.info(f"Loading local LLM: {model_name} ({quantization} quantization)...")
            print(f"\n🔄 Loading local LLM: {model_name}...")
            print(f"   This may take a few minutes on first run (downloading model)")
            
            # Configure quantization
            if quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quantization == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                bnb_config = None
            
            # Load model
            self.local_llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.local_llm_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.llm_provider = "local"
            self.client = "local"  # Flag that we have a working LLM
            self.logger.info(f"Local LLM loaded successfully")
            print(f"✓ Local LLM loaded successfully\n")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize local LLM: {e}")
            print(f"✗ Failed to load local LLM: {e}")
            print(f"  Falling back to rule-based extraction\n")
    
    def _initialize_gemini(self):
        """Initialize Google Gemini API."""
        try:
            from google import genai
            
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.client = client
            self.gemini_model = model_name
            self.llm_provider = "gemini"
            self.logger.info(f"Using Google Gemini (model: {model_name})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Gemini: {e}")
    
    def _initialize_openai(self):
        """Initialize OpenAI API."""
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
            self.llm_provider = "openai"
            self.logger.info(f"Using OpenAI (model: {os.getenv('LLM_MODEL', 'gpt-4o-mini')})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAI: {e}")

        
    async def analyze(self, chunk: TranscriptChunk) -> InsightResult:
        """
        Analyze a transcript chunk and extract insights.
        
        Args:
            chunk: The transcript chunk to analyze
            
        Returns:
            InsightResult with detected insights and updated summary
        """
        import asyncio
        
        # Track conversation history
        self.conversation_history.append(chunk)
        
        # Extract insights
        if self.use_llm and self.client:
            insights = await asyncio.to_thread(self._extract_insights_llm, chunk)
        else:
            insights = self._extract_insights_rule_based(chunk)
        
        # Update rolling summary
        self.current_summary = await asyncio.to_thread(self._update_summary, chunk)
        
        # Store insights
        self.all_insights.extend(insights)
        
        return InsightResult(
            chunk=chunk,
            insights=insights,
            rolling_summary=self.current_summary,
        )
    
    def _extract_insights_llm(self, chunk: TranscriptChunk) -> List[Insight]:
        """
        Extract insights using LLM with structured output (supports local, OpenAI, and Gemini).
        
        Args:
            chunk: The transcript chunk to analyze
            
        Returns:
            List of extracted insights
        """
        try:
            prompt = self._build_prompt(chunk)
            
            if self.llm_provider == "local":
                # Local LLM (Qwen/Llama/etc)
                messages = [
                    {"role": "system", "content": (
                        "You are a financial analyst specializing in Indian earnings calls. "
                        "Extract key insights from conference call transcripts. "
                        "Focus on revenue, guidance, risks, outlook, and financial metrics."
                    )},
                    {"role": "user", "content": prompt}
                ]
                
                # Format as chat template
                text = self.local_llm_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Tokenize and generate
                inputs = self.local_llm_tokenizer([text], return_tensors="pt").to(self.local_llm_model.device)
                max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "500"))
                
                outputs = self.local_llm_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.local_llm_tokenizer.eos_token_id
                )
                
                # Decode response
                content = self.local_llm_tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                
            elif self.llm_provider == "gemini":
                # Google Gemini API
                response = self.client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt
                )
                content = response.text
                
            elif self.llm_provider == "openai":
                # OpenAI API
                response = self.client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a financial analyst specializing in Indian earnings calls. "
                                "Extract key insights from conference call transcripts. "
                                "Focus on revenue, guidance, risks, outlook, and financial metrics."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )
                content = response.choices[0].message.content
                
            else:
                # Fallback if no provider set
                return self._extract_insights_rule_based(chunk)
            
            insights = self._parse_llm_response(content, chunk)
            
            # Add sentiment analysis if available
            if self.sentiment_analyzer:
                insights = self._add_sentiment_to_insights(insights)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}. Falling back to rule-based.")
            return self._extract_insights_rule_based(chunk)
    
    def _build_prompt(self, chunk: TranscriptChunk) -> str:
        """Build the prompt for LLM-based insight extraction."""
        # Get recent context (last 3 chunks)
        recent_context = self.conversation_history[-3:] if len(self.conversation_history) > 1 else []
        context_text = "\n".join([f"[{c.start_time:.0f}s] {c.text}" for c in recent_context[:-1]])
        
        # Build context part separately to avoid f-string nesting issue
        context_part = f"Previous context:\n{context_text}" if context_text else ""
        
        prompt = f"""Analyze this transcript segment from an Indian earnings/conference call:

Current segment: "{chunk.text}"

{context_part}

Extract the following:
1. **Financial Insights**: Any mentions of revenue, growth rates, margins, EBITDA, etc.
2. **Guidance**: Forward-looking statements, projections, targets
3. **Risks**: Challenges, concerns, headwinds mentioned
4. **Outlook**: Future expectations, market conditions
5. **Sentiment**: Overall tone (positive/negative/neutral)

Format your response as:
[INSIGHT_TYPE] | Text | Sentiment
Example:
REVENUE | Revenue grew 15% YoY to 500 crores | positive
GUIDANCE | Targeting 20% growth next quarter | positive
RISK | Supply chain disruptions may impact margins | negative

If no significant insights, respond with: "NO_INSIGHTS"
"""
        return prompt
    
    def _parse_llm_response(self, content: str, chunk: TranscriptChunk) -> List[Insight]:
        """Parse LLM response into Insight objects."""
        insights = []
        
        if "NO_INSIGHTS" in content.upper():
            return insights
        
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("**"):
                continue
            
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                try:
                    insight_type_str = parts[0].upper().strip()
                    text = parts[1].strip()
                    sentiment_str = parts[2].lower().strip() if len(parts) > 2 else "neutral"
                    
                    # Map type
                    insight_type = InsightType.OTHER
                    for it in InsightType:
                        if it.value.upper() in insight_type_str:
                            insight_type = it
                            break
                    
                    # Map sentiment
                    sentiment = Sentiment.NEUTRAL
                    if "positive" in sentiment_str:
                        sentiment = Sentiment.POSITIVE
                    elif "negative" in sentiment_str:
                        sentiment = Sentiment.NEGATIVE
                    
                    insights.append(
                        Insight(
                            type=insight_type,
                            text=text,
                            sentiment=sentiment,
                            source_text=chunk.text,
                            timestamp=chunk.start_time,
                        )
                    )
                except Exception as e:
                    self.logger.debug(f"Failed to parse line: {line} ({e})")
        
        return insights
    
    def _extract_insights_rule_based(self, chunk: TranscriptChunk) -> List[Insight]:
        """
        Fallback rule-based insight extraction.
        
        Simple keyword-based detection for when LLM is unavailable.
        """
        import re
        insights = []
        text = chunk.text.lower()
        
        # Revenue patterns
        revenue_keywords = ['revenue', 'sales', 'turnover', 'topline', 'crore', 'lakh']
        if any(kw in text for kw in revenue_keywords):
            # Check for numbers
            if re.search(r'\d+', text):
                insights.append(Insight(
                    type=InsightType.REVENUE,
                    text=f"Revenue mentioned: {chunk.text}",
                    sentiment=Sentiment.NEUTRAL,
                    source_text=chunk.text,
                    timestamp=chunk.start_time,
                    confidence=1.0,
                ))
        
        # Growth patterns
        growth_keywords = ['growth', 'grew', 'increase', 'expansion', 'yoy', 'qoq']
        if any(kw in text for kw in growth_keywords):
            sentiment = Sentiment.POSITIVE if any(w in text for w in ['strong', 'good', 'excellent']) else Sentiment.NEUTRAL
            insights.append(Insight(
                type=InsightType.GROWTH,
                text=f"Growth mentioned: {chunk.text}",
                sentiment=sentiment,
                source_text=chunk.text,
                timestamp=chunk.start_time,
                confidence=1.0,
            ))
        
        # Risk patterns
        risk_keywords = ['risk', 'concern', 'challenge', 'headwind', 'pressure', 'decline', 'decrease']
        if any(kw in text for kw in risk_keywords):
            insights.append(Insight(
                type=InsightType.RISK,
                text=f"Risk mentioned: {chunk.text}",
                sentiment=Sentiment.NEGATIVE,
                source_text=chunk.text,
                timestamp=chunk.start_time,
                confidence=1.0,
            ))
        
        # Guidance patterns
        guidance_keywords = ['guidance', 'forecast', 'expect', 'outlook', 'target', 'goal']
        if any(kw in text for kw in guidance_keywords):
            insights.append(Insight(
                type=InsightType.GUIDANCE,
                text=f"Guidance provided: {chunk.text}",
                sentiment=Sentiment.NEUTRAL,
                source_text=chunk.text,
                timestamp=chunk.start_time,
                confidence=1.0,
            ))
        
        # Add sentiment analysis if available
        if self.sentiment_analyzer and insights:
            insights = self._add_sentiment_to_insights(insights)
        
        return insights
    
    def _add_sentiment_to_insights(self, insights: List[Insight]) -> List[Insight]:
        """
        Add sentiment analysis to insights using transformers pipeline.
        
        Args:
            insights: List of insights to analyze
            
        Returns:
            Updated insights with ML-based sentiment
        """
        if not self.sentiment_analyzer:
            return insights
        
        for insight in insights:
            try:
                # Run sentiment analysis
                result = self.sentiment_analyzer(insight.text[:512])[0]  # Max 512 chars
                label = result['label'].lower()
                score = result['score']
                
                # Map to our sentiment enum with confidence threshold
                # If confidence is low (<0.6), treat as neutral
                if score < 0.6:
                    insight.sentiment = Sentiment.NEUTRAL
                    insight.confidence = score
                elif 'positive' in label:
                    insight.sentiment = Sentiment.POSITIVE
                    insight.confidence = score
                elif 'negative' in label:
                    insight.sentiment = Sentiment.NEGATIVE
                    insight.confidence = score
                else:
                    insight.sentiment = Sentiment.NEUTRAL
                    insight.confidence = score
                    
            except Exception as e:
                self.logger.debug(f"Sentiment analysis failed for insight: {e}")
        
        return insights
    
    def _update_summary(self, chunk: TranscriptChunk) -> str:
        """Update the rolling summary with new chunk information."""
        # Keep a concise summary of the last few chunks
        recent_chunks = self.conversation_history[-5:]  # Last 5 chunks
        
        # Simple concatenation for now (LLM would be better for this)
        if len(recent_chunks) <= 2:
            return chunk.text
        else:
            # Create a brief summary
            summary_parts = []
            for c in recent_chunks[-3:]:
                # Take first sentence or 100 chars
                first_sentence = c.text.split('.')[0] if '.' in c.text else c.text[:100]
                summary_parts.append(first_sentence.strip())
            
            return " ... ".join(summary_parts)
    
    def get_final_summary(self) -> str:
        """
        Generate a final summary of the entire call.
        
        Call this after processing all chunks.
        """
        if not self.conversation_history:
            return "No transcript available."
        
        # Combine all insights into a summary
        summary_parts = []
        
        # Group insights by type
        insights_by_type = {}
        for insight in self.all_insights:
            if insight.type not in insights_by_type:
                insights_by_type[insight.type] = []
            insights_by_type[insight.type].append(insight)
        
        # Build summary
        if InsightType.REVENUE in insights_by_type:
            summary_parts.append(f"Revenue/Financial: {len(insights_by_type[InsightType.REVENUE])} mentions")
        
        if InsightType.GROWTH in insights_by_type:
            summary_parts.append(f"Growth: {len(insights_by_type[InsightType.GROWTH])} mentions")
        
        if InsightType.GUIDANCE in insights_by_type:
            summary_parts.append(f"Guidance: {len(insights_by_type[InsightType.GUIDANCE])} mentions")
        
        if InsightType.RISK in insights_by_type:
            summary_parts.append(f"Risks: {len(insights_by_type[InsightType.RISK])} mentions")
        
        # Overall sentiment
        positive = sum(1 for i in self.all_insights if i.sentiment == Sentiment.POSITIVE)
        negative = sum(1 for i in self.all_insights if i.sentiment == Sentiment.NEGATIVE)
        
        if positive > negative:
            sentiment_summary = "Overall sentiment: POSITIVE"
        elif negative > positive:
            sentiment_summary = "Overall sentiment: NEGATIVE"
        else:
            sentiment_summary = "Overall sentiment: NEUTRAL"
        
        summary = f"Call Summary: {', '.join(summary_parts)}. {sentiment_summary}. Total insights: {len(self.all_insights)}"
        
        return summary
    
    def get_key_insights(self) -> List[Insight]:
        """Get the most important insights detected so far."""
        return self.all_insights

