# core/llm/kimi_client.py
"""
Kimi (Moonshot) API Client for NEXUS AI.

Checkpoint: 7.1 - Updated for Kimi K2.5

Official Moonshot API documentation: https://platform.moonshot.ai/docs

Kimi K2.5 provides:
- Native multimodal agentic capabilities
- Thinking mode with reasoning traces
- Agent Swarm for complex task decomposition
- Cost-effective LLM for match analysis

Uses OpenAI-compatible API format.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class KimiModel(str, Enum):
    """
    Available Kimi models on Moonshot AI platform.

    Reference: https://platform.moonshot.ai/docs
    """
    # Kimi K2.5 models (latest, recommended)
    K2_5_PREVIEW = "kimi-k2.5-preview"           # Latest K2.5 multimodal agentic model

    # Kimi K2 models
    K2_THINKING = "kimi-k2-thinking"             # K2 with deep reasoning (CoT), recommended temp=1.0
    K2_0905_PREVIEW = "kimi-k2-0905-preview"     # September 2025 version
    K2_0711_PREVIEW = "kimi-k2-0711-preview"     # July 2025 version

    # Legacy Moonshot V1 models (still available)
    V1_8K = "moonshot-v1-8k"                     # 8k context, fastest
    V1_32K = "moonshot-v1-32k"                   # 32k context, balanced
    V1_128K = "moonshot-v1-128k"                 # 128k context, longest context


class KimiMode(str, Enum):
    """Operating modes for Kimi K2.5."""
    THINKING = "thinking"  # Includes reasoning traces, temp=1.0 recommended
    INSTANT = "instant"    # Direct responses, temp=0.6 recommended


@dataclass
class KimiResponse:
    """Response from Kimi API."""
    success: bool
    content: str
    model: str
    usage: Dict[str, int]
    error: Optional[str] = None
    reasoning_content: Optional[str] = None  # For thinking mode

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def has_reasoning(self) -> bool:
        """Check if response includes reasoning traces."""
        return bool(self.reasoning_content)


class KimiClient:
    """
    Async client for Kimi (Moonshot) API.

    Official API endpoint: https://api.moonshot.ai/v1

    Features:
    - OpenAI-compatible API format
    - Thinking mode with reasoning traces
    - Agent capabilities for complex tasks
    - Multiple model variants (K2.5, K2, V1)

    Usage:
        async with KimiClient() as client:
            # Instant mode (fast, direct)
            response = await client.chat("Analyze this match...", mode=KimiMode.INSTANT)

            # Thinking mode (deep reasoning)
            response = await client.chat_thinking("Complex analysis...")
            print(response.reasoning_content)  # Reasoning traces
            print(response.content)  # Final answer
    """

    # Official Moonshot API endpoint
    BASE_URL = "https://api.moonshot.ai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = KimiModel.K2_5_PREVIEW,
        timeout: float = 120.0,  # Increased for thinking mode
        max_retries: int = 3,
    ):
        """
        Initialize Kimi client.

        Args:
            api_key: Moonshot API key (MOONSHOT_API_KEY). If None, reads from settings.
            model: Model to use. Default is kimi-k2.5-preview (latest).
            timeout: Request timeout in seconds (higher for thinking mode)
            max_retries: Number of retries on failure
        """
        self.api_key = api_key if api_key is not None else self._get_api_key()
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    def _get_api_key(self) -> Optional[str]:
        """Get API key from settings (MOONSHOT_API_KEY)."""
        try:
            from config.settings import settings
            return settings.MOONSHOT_API_KEY
        except Exception:
            return None

    @property
    def is_available(self) -> bool:
        """Check if Moonshot API is configured."""
        return bool(self.api_key) and not self._is_placeholder(self.api_key)

    def _is_placeholder(self, value: str) -> bool:
        """Check if value is a placeholder."""
        placeholders = ["your_", "placeholder", "xxx", "change_me", "none", "null"]
        return any(p in value.lower() for p in placeholders)

    async def __aenter__(self) -> "KimiClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        mode: KimiMode = KimiMode.INSTANT,
    ) -> KimiResponse:
        """
        Send a chat message to Kimi.

        Args:
            message: User message
            system_prompt: Optional system prompt
            temperature: Sampling temperature. If None, uses mode default:
                         - INSTANT: 0.6
                         - THINKING: 1.0
            max_tokens: Maximum tokens in response
            model: Override default model
            mode: INSTANT (fast) or THINKING (with reasoning)

        Returns:
            KimiResponse with content and optional reasoning_content
        """
        if not self.is_available:
            return KimiResponse(
                success=False,
                content="",
                model=self.model,
                usage={},
                error="Moonshot API key not configured (MOONSHOT_API_KEY)"
            )

        # Set default temperature based on mode
        if temperature is None:
            temperature = 1.0 if mode == KimiMode.THINKING else 0.6

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        return await self._send_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model or self.model,
        )

    async def chat_thinking(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 8192,
        model: Optional[str] = None,
    ) -> KimiResponse:
        """
        Send a chat message using thinking mode with reasoning traces.

        Uses K2 thinking model with temperature=1.0 for best reasoning.
        Response includes reasoning_content field with chain-of-thought.

        Args:
            message: User message
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens (higher for thinking)
            model: Override model (defaults to kimi-k2-thinking)

        Returns:
            KimiResponse with content and reasoning_content
        """
        thinking_model = model or KimiModel.K2_THINKING

        return await self.chat(
            message=message,
            system_prompt=system_prompt,
            temperature=1.0,  # Recommended for thinking mode
            max_tokens=max_tokens,
            model=thinking_model,
            mode=KimiMode.THINKING,
        )

    async def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> KimiResponse:
        """
        Send chat with conversation history.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            temperature: Sampling temperature (0.6 for instant, 1.0 for thinking)
            max_tokens: Maximum tokens in response
            model: Override default model

        Returns:
            KimiResponse
        """
        if not self.is_available:
            return KimiResponse(
                success=False,
                content="",
                model=self.model,
                usage={},
                error="Moonshot API key not configured (MOONSHOT_API_KEY)"
            )

        return await self._send_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model or self.model,
        )

    async def _send_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> KimiResponse:
        """Send request to Moonshot API with retries."""
        if not self._client:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            ) as client:
                return await self._do_request(
                    client, messages, temperature, max_tokens, model
                )

        return await self._do_request(
            self._client, messages, temperature, max_tokens, model
        )

    async def _do_request(
        self,
        client: httpx.AsyncClient,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> KimiResponse:
        """Execute the API request with retry logic."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    json=payload,
                )

                if response.status_code == 200:
                    data = response.json()
                    message_data = data["choices"][0]["message"]

                    # Extract reasoning_content for thinking mode
                    reasoning_content = message_data.get("reasoning_content")

                    return KimiResponse(
                        success=True,
                        content=message_data.get("content", ""),
                        model=data.get("model", model),
                        usage=data.get("usage", {}),
                        reasoning_content=reasoning_content,
                    )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"Moonshot rate limited, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue

                # Handle other errors
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Moonshot API error: {error_msg}")
                last_error = error_msg

            except httpx.TimeoutException as e:
                last_error = f"Request timeout: {e}"
                logger.warning(f"Moonshot timeout (attempt {attempt + 1}): {e}")

            except Exception as e:
                last_error = str(e)
                logger.error(f"Moonshot request failed (attempt {attempt + 1}): {e}")

            # Exponential backoff
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        return KimiResponse(
            success=False,
            content="",
            model=model,
            usage={},
            error=last_error or "Max retries exceeded",
        )

    # =========================================================================
    # Sports Analysis Methods
    # =========================================================================

    async def analyze_match(
        self,
        home_team: str,
        away_team: str,
        league: str,
        context: Optional[str] = None,
        use_thinking: bool = True,
    ) -> KimiResponse:
        """
        Analyze a match for betting insights using K2.5 reasoning.

        Args:
            home_team: Home team name
            away_team: Away team name
            league: League/competition name
            context: Additional context (recent form, injuries, news)
            use_thinking: Use thinking mode for deeper analysis

        Returns:
            KimiResponse with analysis
        """
        system_prompt = """You are a professional sports analyst specializing in betting markets.
Analyze matches objectively, considering:
- Team form and momentum
- Head-to-head records
- Home/away performance
- Key player availability
- Tactical matchups
- Market value indicators

Provide concise, actionable insights with confidence levels (1-5).
Format: JSON with keys: analysis, prediction, confidence, key_factors, risks"""

        message = f"""Analyze this {league} match:
{home_team} (Home) vs {away_team} (Away)

{f"Additional context: {context}" if context else ""}

Provide analysis in JSON format."""

        if use_thinking:
            return await self.chat_thinking(
                message=message,
                system_prompt=system_prompt,
                max_tokens=2048,
            )
        else:
            return await self.chat(
                message=message,
                system_prompt=system_prompt,
                temperature=0.6,
                max_tokens=1024,
                mode=KimiMode.INSTANT,
            )

    async def extract_injuries(
        self,
        news_text: str,
        team_names: Optional[List[str]] = None,
    ) -> KimiResponse:
        """
        Extract injury information from news text.

        Args:
            news_text: News article or text containing injury info
            team_names: Optional list of team names to focus on

        Returns:
            KimiResponse with extracted injuries in JSON format
        """
        system_prompt = """Extract injury information from sports news.
Return JSON with:
{
    "injuries": [
        {
            "player": "Player Name",
            "team": "Team Name",
            "injury_type": "type of injury",
            "status": "out|doubtful|questionable|probable",
            "expected_return": "date or duration if mentioned",
            "source_quote": "relevant quote from text"
        }
    ],
    "suspensions": [
        {
            "player": "Player Name",
            "team": "Team Name",
            "reason": "reason",
            "matches_remaining": number
        }
    ]
}

If no injuries found, return empty arrays."""

        team_filter = ""
        if team_names:
            team_filter = f"\nFocus on these teams: {', '.join(team_names)}"

        message = f"""Extract injury and suspension information from this text:{team_filter}

---
{news_text}
---

Return JSON only."""

        return await self.chat(
            message=message,
            system_prompt=system_prompt,
            temperature=0.3,  # Low temp for extraction accuracy
            max_tokens=1024,
            mode=KimiMode.INSTANT,
        )

    async def generate_reasoning(
        self,
        ml_prediction: Dict[str, Any],
        match_context: Dict[str, Any],
    ) -> KimiResponse:
        """
        Generate reasoning to enhance ML prediction using K2.5 thinking mode.

        Args:
            ml_prediction: ML model prediction (probabilities, expected goals, etc.)
            match_context: Context including teams, form, injuries, odds

        Returns:
            KimiResponse with reasoning_content and adjusted prediction
        """
        system_prompt = """You are an AI sports betting assistant that combines ML predictions with contextual reasoning.

Given an ML model's prediction and match context, provide:
1. Analysis of factors the ML might have missed
2. Confidence adjustment based on context
3. Final recommendation

Return JSON:
{
    "ml_analysis": "assessment of ML prediction quality",
    "contextual_factors": ["factor1", "factor2", ...],
    "confidence_adjustment": float (-0.2 to +0.2),
    "final_confidence": float (0.0 to 1.0),
    "recommendation": "bet type and reasoning",
    "risk_level": "low|medium|high",
    "key_insight": "one sentence summary"
}"""

        message = f"""Combine ML prediction with context:

ML Prediction:
{self._format_dict(ml_prediction)}

Match Context:
{self._format_dict(match_context)}

Provide reasoning and adjusted prediction in JSON."""

        # Use thinking mode for deeper reasoning
        return await self.chat_thinking(
            message=message,
            system_prompt=system_prompt,
            max_tokens=2048,
        )

    def _format_dict(self, d: Dict[str, Any], indent: int = 2) -> str:
        """Format dictionary for prompt."""
        import json
        return json.dumps(d, indent=indent, ensure_ascii=False, default=str)


# =============================================================================
# Agent Swarm Support (K2.5 Feature)
# =============================================================================

@dataclass
class AgentTask:
    """A sub-task for agent swarm execution."""
    task_id: str
    description: str
    agent_type: str
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[str] = None


class KimiAgentSwarm:
    """
    Agent Swarm orchestrator using Kimi K2.5.

    K2.5 transitions from single-agent scaling to a self-directed,
    coordinated swarm-like execution scheme. It decomposes complex
    tasks into parallel sub-tasks executed by dynamically instantiated,
    domain-specific agents.

    Usage:
        swarm = KimiAgentSwarm()
        result = await swarm.execute_complex_task(
            "Analyze all Premier League matches this weekend with injuries,
             news, form analysis, and betting recommendations"
        )
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = KimiClient(
            api_key=api_key,
            model=KimiModel.K2_5_PREVIEW,
            timeout=180.0,  # Higher timeout for swarm operations
        )
        self.tasks: List[AgentTask] = []

    async def decompose_task(self, complex_task: str) -> List[AgentTask]:
        """
        Use K2.5 to decompose a complex task into sub-tasks.

        Args:
            complex_task: Description of the complex task

        Returns:
            List of AgentTask objects
        """
        system_prompt = """You are a task decomposition agent. Break down complex tasks into smaller,
parallelizable sub-tasks. For each sub-task, specify:
- task_id: unique identifier
- description: what needs to be done
- agent_type: type of specialist agent needed (data_collector, analyzer, predictor, reporter)

Return JSON array of tasks."""

        message = f"""Decompose this complex task into parallel sub-tasks:

{complex_task}

Return JSON array of tasks."""

        async with self.client as client:
            response = await client.chat(
                message=message,
                system_prompt=system_prompt,
                temperature=0.6,
                mode=KimiMode.INSTANT,
            )

        if response.success:
            import json
            try:
                tasks_data = json.loads(self._extract_json(response.content))
                self.tasks = [
                    AgentTask(
                        task_id=t.get("task_id", f"task_{i}"),
                        description=t.get("description", ""),
                        agent_type=t.get("agent_type", "general"),
                    )
                    for i, t in enumerate(tasks_data)
                ]
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse task decomposition")
                self.tasks = []

        return self.tasks

    async def execute_task(self, task: AgentTask) -> AgentTask:
        """Execute a single agent task."""
        task.status = "running"

        agent_prompts = {
            "data_collector": "You are a data collection agent. Gather relevant information.",
            "analyzer": "You are an analysis agent. Provide deep insights.",
            "predictor": "You are a prediction agent. Make data-driven predictions.",
            "reporter": "You are a reporting agent. Summarize findings clearly.",
        }

        system_prompt = agent_prompts.get(
            task.agent_type,
            "You are a general-purpose agent. Complete the task efficiently."
        )

        async with self.client as client:
            response = await client.chat_thinking(
                message=task.description,
                system_prompt=system_prompt,
            )

        if response.success:
            task.status = "completed"
            task.result = response.content
        else:
            task.status = "failed"
            task.result = response.error

        return task

    async def execute_complex_task(
        self,
        complex_task: str,
        max_parallel: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute a complex task using agent swarm.

        Args:
            complex_task: The complex task to execute
            max_parallel: Maximum parallel agents

        Returns:
            Combined results from all agents
        """
        # Decompose task
        tasks = await self.decompose_task(complex_task)

        if not tasks:
            return {"error": "Failed to decompose task", "tasks": []}

        # Execute tasks with semaphore for parallelism control
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_with_semaphore(task: AgentTask) -> AgentTask:
            async with semaphore:
                return await self.execute_task(task)

        # Run all tasks in parallel
        completed_tasks = await asyncio.gather(
            *[execute_with_semaphore(t) for t in tasks],
            return_exceptions=True,
        )

        # Compile results
        results = {
            "task_count": len(tasks),
            "completed": sum(1 for t in completed_tasks if isinstance(t, AgentTask) and t.status == "completed"),
            "failed": sum(1 for t in completed_tasks if isinstance(t, AgentTask) and t.status == "failed"),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "description": t.description,
                    "agent_type": t.agent_type,
                    "status": t.status,
                    "result": t.result,
                }
                for t in completed_tasks
                if isinstance(t, AgentTask)
            ],
        }

        return results

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown."""
        import re

        # Try to find JSON array or object
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'(\[[\s\S]*\])',
            r'(\{[\s\S]*\})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)

        return text


# =============================================================================
# Convenience Functions
# =============================================================================

async def get_kimi_analysis(
    home_team: str,
    away_team: str,
    league: str,
    context: Optional[str] = None,
    api_key: Optional[str] = None,
    use_thinking: bool = True,
) -> KimiResponse:
    """
    Quick function to get match analysis from Kimi K2.5.

    Usage:
        response = await get_kimi_analysis("Arsenal", "Chelsea", "Premier League")
        if response.success:
            print(response.reasoning_content)  # Thinking traces
            print(response.content)  # Final analysis
    """
    async with KimiClient(api_key=api_key) as client:
        return await client.analyze_match(
            home_team=home_team,
            away_team=away_team,
            league=league,
            context=context,
            use_thinking=use_thinking,
        )


async def get_swarm_analysis(
    task: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quick function to run agent swarm analysis.

    Usage:
        result = await get_swarm_analysis(
            "Analyze all Premier League matches with full context"
        )
    """
    swarm = KimiAgentSwarm(api_key=api_key)
    return await swarm.execute_complex_task(task)
