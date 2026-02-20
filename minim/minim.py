#!/usr/bin/env python3
from types import CoroutineType
from datetime import datetime
from pydantic import BaseModel
from typing import Literal, Tuple, Callable, Awaitable, Any, TypeVar
import logging
import asyncio
from forecasting_tools import (
    structure_output,
    clean_indents,
    ForecastBot,
    GeneralLlm,
    MetaculusQuestion,
    SpringTemplateBot2026,
    BinaryQuestion,
    ConditionalQuestion,
    DateQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    ReasonedPrediction,
    PredictedOptionList,
    NumericDistribution,
)


from minim.researcher import MinimResearcher

logger = logging.getLogger(__name__)

T_Question = TypeVar("T_Question", bound="MetaculusQuestion")

ReasoningError = Literal["NONE", "TIME", "BASE RATE", "OTHER"]


class ReasoningCheck(BaseModel):
    error_type: ReasoningError
    error_explanation: str


class Minim(SpringTemplateBot2026):
    """
    This is a minimally modified bot for the Metaculus AI forecasting tournaments. In particular, the present modification concerns the procedure to research a question and a check for logic issues in returned reasoning.
    Instead of calling a research API directly, this bot passes off the work to a dedicated researcher class, MinimResearcher. See that class for details of the research procedure.
    After a forecast has been made, it is additionally checked for some logical errors which occurred in testing.
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

    async def _validate_reasoning(
        self, question: MetaculusQuestion, prediction: ReasonedPrediction
    ) -> ReasoningCheck:
        prompt = clean_indents(f"""
            You are a component of a forecasting system.
            The forecasting system generally produces high-quality predictions, but occasionally reasons poorly.
            Your job is to check the reasoning of a prediction and ensure that it logically coheres and doesn't fall into any of a few categories of mistakes.
            You do not produce forecasts yourself.
            You receive a question, its deciding criteria, and the system's reasoning for its prediction. You decide whether it fits into any of the categories of error given, or whether it has some other serious logical error which compromises the validity of the prediction. You do not evaluate how good the prediction is, just whether the reasoning is logically valid. 
            The categories of errors you are looking out for are as follows:
            1. Errors of event timing. Sometimes, the question to forecast will be about something which is revealed at a later date than it is determined. For instance, a question might be "How well will AI do in this 2026 prediction contest?" and predictions are registered in January 2026, then scored in December 2026. It is logically incorrect to consider improvements in AI technology over the course of 2026; the only thing that can affect the results of the contest is how the technology stands in January 2026. Likewise, if a question asks "What will be the reported income of Apple in its December 2025 earnings report?" and the report will be released in February 2026, it would be an error to consider changes in income that might occur in January 2026, since they could not affect the report.
            2. Errors of considering the "status quo" to be something other than the base rate. The system is biased towards the "status quo" outcome since the world changes slowly most of the time. However, it sometimes makes an error in determining what the status quo is. For instance, if recently the WHO has reported a public health emergency in one out of three years, it is a logical mistake to reason that the "status quo" is no public health emergency and to thus predict a public health emergency with a chance of less than one in three. The status quo that should be biased towards is the recent base rate, not anything else. This category does not apply to any reasoning which does not explicitly cite the "nothing happens"/"status quo" bias as a reason to predict something other than the base rate.

            The question that the system has forecast is:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}

            The reasoning of the forecasting system is:
            {prediction.reasoning}

            You decide whether this reasoning has a logical error of timing or of base rate neglect, or any other serious logical error. You do not try to determine whether the reasoning is correct in any other respect, or whether the prediction is correct; you just try to detect logical errors.
            You first write your reasoning for why the system's reasoning is logically invalid.
            Next, on its own line, you write "FINAL ANSWER: " followed by "NONE" if there is no serious logical error, "TIME" if there is an error of event timing, "BASE RATE" if there is an error of base rate neglect, or "OTHER" if there is a different logical error.
            Finally, if your answer wasn't NONE, you explain what the error was on the next line. You do not need to explain anything if you answer NONE.
        """)

        reasoning = await self.get_llm("default", "llm").invoke(prompt)

        reasoningerror: ReasoningError = await structure_output(
            reasoning,
            ReasoningError,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )

        if reasoningerror == "NONE":
            return ReasoningCheck(error_type=reasoningerror, error_explanation="")
        else:
            i = reasoning.casefold().find("final answer:")
            i = reasoning.find("\n")
            return ReasoningCheck(
                error_type=reasoningerror, error_explanation=reasoning[i:]
            )

    async def _run_forecast_with_checking(
        self,
        question: T_Question,
        prompt_head: str,
        prompt_tail: str,
        forecast_fun: Callable[[T_Question, str], Awaitable[ReasonedPrediction]],
    ) -> ReasonedPrediction:

        prediction = await forecast_fun(question, "\n".join([prompt_head, prompt_tail]))

        reasoningcheck = await self._validate_reasoning(question, prediction)

        if reasoningcheck.error_type != "NONE":
            if reasoningcheck.error_type == "OTHER":
                logger.warning(
                    f"Logic checker found an uncategorised error: \n{reasoningcheck.error_explanation}"
                )
            else:
                logger.info(
                    f"Logic checker found an error of type {reasoningcheck.error_type}"
                )
            prompt_error = clean_indents(f"""
             A previous attempt at forecasting this question had a logical error. This error was:
             {reasoningcheck.error_explanation}
             You should make sure to avoid this error when forecasting.
             """)

            prediction = await forecast_fun(
                question, "\n".join([prompt_head, prompt_error, prompt_tail])
            )

        return prediction

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:

        prompt_head = clean_indents(f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}
            """)

        prompt_tail = clean_indents("""
        The last thing you write is your final answer as: "Probability: ZZ%", 0-100
        """)

        return await self._run_forecast_with_checking(
            question=question,
            prompt_head=prompt_head,
            prompt_tail=prompt_tail,
            forecast_fun=self._binary_prompt_to_forecast,
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt_head = clean_indents(f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.
""")

        prompt_tail = clean_indents(f"""
            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """)

        return await self._run_forecast_with_checking(
            question=question,
            prompt_head=prompt_head,
            prompt_tail=prompt_tail,
            forecast_fun=self._multiple_choice_prompt_to_forecast,
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt_head = clean_indents(f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            """)
        prompt_tail = clean_indents(f"""

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """)

        return await self._run_forecast_with_checking(
            question=question,
            prompt_head=prompt_head,
            prompt_tail=prompt_tail,
            forecast_fun=self._numeric_prompt_to_forecast,
        )

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt_head = clean_indents(f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ.No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            """)

        prompt_tail = clean_indents(f"""

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """)

        return await self._run_forecast_with_checking(
            question=question,
            prompt_head=prompt_head,
            prompt_tail=prompt_tail,
            forecast_fun=self._date_prompt_to_forecast,
        )
