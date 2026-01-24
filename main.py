from litellm import num_retries
import argparse
import asyncio
from aiolimiter import AsyncLimiter
import logging
import dotenv
from typing import Literal

from forecasting_tools import (
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    ForecastReport,
)

from minim.researcher import MinimResearcher
from minim.minim import Minim
from minim.ratelimiter import RateLimitedLlm


dotenv.load_dotenv()
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run the minim forecasting system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        # default="tournament",
        help="Specify the run mode",  # (default: tournament)",
    )
    parser.add_argument(
        "--reportdir",
        type=str,
        default=None,
        help="Specify the folder to save reports to",
    )
    parser.add_argument(
        "--researchdir",
        type=str,
        default=None,
        help="Specify the folder to save research to",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"
    reports_dir: str | None = args.reportdir
    research_dir: str | None = args.researchdir

    reasoner = RateLimitedLlm(
        model="openrouter/openai/gpt-5.2",
        rate_limiter=AsyncLimiter(500),
        reasoning_effort="high",
        temperature=0.3,
        timeout=15 * 60,  # settings should be from metac-gpt5.2-high
    )
    minimodel = GeneralLlm(model="openrouter/openai/gpt-4o-mini")
    asknews_researcher = "asknews/news-summaries"

    researcher = MinimResearcher(
        parser=minimodel,
        general_model=minimodel,
        asknews_researcher=asknews_researcher,
        report_dir=research_dir,
    )

    minim_bot = Minim(
        researcher=researcher,
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=reports_dir,  # folder is created if it i
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": RateLimitedLlm(
                model="openrouter/openai/gpt-5.2",
                rate_limiter=AsyncLimiter(500),
                reasoning_effort="high",
                temperature=0.3,
                timeout=15 * 60,  # settings should be from metac-gpt5.2-high
            ),
            "summarizer": minimodel,
        },
    )

    client = MetaculusClient()
    if run_mode == "tournament":
        # You may want to change this to the specific tournament ID you want to forecast on
        seasonal_tournament_reports = asyncio.run(
            minim_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            minim_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        minim_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            minim_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            # "https://www.metaculus.com/questions/41131/top-arc-agi-2-score-in-2026/",  # A question with many irrelevant news reports
            "https://www.metaculus.com/questions/41342/",  # A question with some subquestions
            # "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            # "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            # "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            # "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        minim_bot.skip_previously_forecasted_questions = False
        minim_bot.publish_reports_to_metaculus = False
        questions = [
            client.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            minim_bot.forecast_questions(questions, return_exceptions=True)
        )
    minim_bot.log_report_summary(forecast_reports)
