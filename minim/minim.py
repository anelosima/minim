#!/usr/bin/env python3
import logging
import asyncio
from forecasting_tools import (
    ForecastBot,
    GeneralLlm,
    MetaculusQuestion,
    SpringTemplateBot2026,
)

from minim.researcher import MinimResearcher

logger = logging.getLogger(__name__)


class Minim(SpringTemplateBot2026):
    """
    This is a minimally modified bot for the Metaculus AI forecasting tournaments. In particular, the present modification concerns the procedure to research a question; instead of calling a research API directly, this bot passes off the work to a dedicated researcher class, MinimResearcher. See that class for details.
    """

    _structure_output_validation_samples = 2

    def __init__(
        self,
        *,
        researcher: MinimResearcher,
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
        SpringTemplateBot2026.__init__(
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

        self.researcher = researcher

    async def run_research(self, question: MetaculusQuestion) -> str:
        research = await self.researcher.run_research(question)
        logger.info(f"Found Research for URL {question.page_url}:\n{research}")
        return research
