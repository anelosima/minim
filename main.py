import argparse
import asyncio
import logging
import dotenv
from typing import Literal

from forecasting_tools import (
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    clean_indents,
    SpringTemplateBot2026,
)

from minim.researcher import MinimResearcher

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class Minim(SpringTemplateBot2026):
    """
    This is a minimally modified bot for the Metaculus AI forecasting tournaments. In particular, the present modification concerns the procedure to research a question; instead of calling a research API directly, this bot passes off the work to a dedicated researcher class, MinimResearcher. See that class for details.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def __init__(
        self,
        *,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        llms: (
            dict[str, str | GeneralLlm | None] | None
        ) = None,  # Default LLMs are used if llms is set to None
        enable_summarize_research: bool = True,
        parameters_to_exclude_from_config_dict: list[str] | None = None,
        extra_metadata_in_explanation: bool = False,
        required_successful_predictions: float = 0.5,
    ) -> None:
        ForecastBot.__init__(
            self,
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            llms=llms,
            enable_summarize_research=enable_summarize_research,
            parameters_to_exclude_from_config_dict=parameters_to_exclude_from_config_dict,
            extra_metadata_in_explanation=extra_metadata_in_explanation,
            required_successful_predictions=required_successful_predictions,
        )

        self.researcher = MinimResearcher(
            parser=self.get_llm("parser", "llm"),
            relevance_checker=self.get_llm("relevance_checker", "llm"),
            asknews_researcher=self.get_llm("asknews_researcher", "string_name"),
        )  # currently this means that any initialisation without a relevance_checker and an asknews_researcher will fail; they have no defaults

    async def run_research(self, question: MetaculusQuestion) -> str:
        research = await self.researcher.run_research(question)
        logger.info(f"Found Research for URL {question.page_url}:\n{research}")
        return research


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run the Minim forecasting system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        # default="tournament",
        help="Specify the run mode",  # (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    minim_bot = Minim(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to="tests",  # None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": GeneralLlm(  # settings should be from metac-o3
                model="o3",  # o3 is the best reasoning LLM based on the metric of verified performance which is from a company offering API credits for use in the AI benchmark. The alternative would be the latest Gemini.
                temperature=1,
                reasoning_effort="medium",
                timeout=60 * 8,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "asknews_researcher": "asknews/news-summaries",
            "relevance_checker": "openai/gpt-4o-mini",
            "parser": "openai/gpt-4o-mini",
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
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            # "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            # "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            # "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        minim_bot.skip_previously_forecasted_questions = False
        questions = [
            client.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            minim_bot.forecast_questions(questions, return_exceptions=True)
        )
    minim_bot.log_report_summary(forecast_reports)
