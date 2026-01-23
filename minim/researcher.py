from faker.sphinx.documentor import _write
from sklearn.neighbors.tests.test_neighbors import test_k_and_radius_neighbors_X_None
import os
import asyncio
import logging
from datetime import datetime
import time
import json
from typing import List, Optional
from aiolimiter import AsyncLimiter

try:
    from asknews_sdk.dto.news import SearchResponseDictItem
except ImportError:
    pass
try:
    from asknews_sdk.dto.deepnews import CreateDeepNewsResponse
except ImportError:
    pass
try:
    from asknews_sdk import AsyncAskNewsSDK
except ImportError:
    pass

try:
    from asknews_sdk.dto.base import Article
except ImportError:
    pass

from pydantic import BaseModel

from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.util import file_manipulation

from forecasting_tools import (
    structure_output,
    clean_indents,
    AskNewsSearcher,
    GeneralLlm,
    MetaculusQuestion,
)

from minim.ratelimiter import UnboundedAsyncLimiter

logger = logging.getLogger(__name__)


class TimestampedAskNewsSearch(BaseModel):
    timestamp: float
    report: List[SearchResponseDictItem]


class MinimResearcher:
    """
    This is the researcher for the minim forecasting bot. TODO: add details
    """

    _asknews_rate_limit = 12.0

    def __init__(
        self,
        parser: GeneralLlm,
        relevance_checker: GeneralLlm,
        asknews_researcher: str,
        report_dir: str | None = None,
    ):
        self.parser = parser
        self.relevance_checker = relevance_checker
        self.asknews_researcher = asknews_researcher
        self.asknews_limiter = UnboundedAsyncLimiter(1, self._asknews_rate_limit)
        self.report_dir = report_dir

    async def run_research(self, question: MetaculusQuestion) -> str:
        research = ""

        asknewsquery = (
            question.question_text
        )  # NOTE: the template bot includes a full prompt here. Past template bots, which have done well, have not; motivated by a desire not to fix what isn't broken, I've reverted back to including just the question. THIS CAUSES PROBLEMS, but they have empirically not been enough to make the bot not work.
        asknewsresearch = await MinimAskNewsSearcher(
            parser=self.parser,
            relevance_checker=self.relevance_checker,
            question=question,
            rate_limiter=self.asknews_limiter,
            report_dir=self.report_dir,
        ).call_preconfigured_version(self.asknews_researcher, asknewsquery)

        return asknewsresearch


# this fairly ugly structure is necessary to reuse the code for the AskNewsSearcher which appears to work very well
class MinimAskNewsSearcher(AskNewsSearcher):
    """
    This is a modification of the AskNewsSearcher in forecast_tools which should omit irrelevant articles.
    """

    freshness_threshold_days = 7

    def __init__(
        self,
        *,
        parser: GeneralLlm,
        relevance_checker: GeneralLlm,
        question: MetaculusQuestion,
        rate_limiter: UnboundedAsyncLimiter,
        report_dir: str | None = None,
    ):
        AskNewsSearcher.__init__(self)
        self.parser = parser
        self.relevance_checker = relevance_checker
        self.question = question
        self.rate_limiter = rate_limiter
        self.report_dir = report_dir

    async def check_summary(self, query: str, article: Article) -> bool:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster gives you a question they intend to forecast on, and a researcher gives you an article that they have found that may be relevant to the forecast of the question.
            You decide whether the article is relevant in any way to the question being forecast, so as to promote only relevant articles to the attention of the superforecaster.
            You do not make forecasts or do research yourself.
            Articles may be relevant in non-obvious ways. For instance, if the superforecaster is determining whether an event will happen in the future, similar events which happened in the past may be relevant, even if they are not the exact same kind of event.
            You should always err on the side of declaring an article relevant; the superforecaster is very good at disregarding irrelevant information. It is much more important that they have all relevant information than that they never receive irrelevant information.

            The question the superforecaster gives you is:
            {self.question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {self.question.resolution_criteria}
            
            {self.question.fine_print}

            The researcher gives you the following article:
            
            **{article.eng_title}**
            {article.summary}

            You need to decide whether this article is relevant to the forecasting of the question or not.
            You first write your reasoning, explaining why the article is relevant or why it is not.
            Then, to finish your response, you write a line consisting of "Final answer: " followed by the word "true" or the word "false". You write "true" when the article is relevant, and "false" when it is not.
            """
        )

        response = await self.relevance_checker.invoke(prompt)
        relevant: bool = await structure_output(
            response,
            bool,
            model=self.parser,
            num_validation_samples=2,
        )
        logger.info(
            f'Article with headline "{article.eng_title}" deemed '
            + ("relevant" if relevant else "irrelevant")
            + "."
        )
        return relevant

    async def get_formatted_news_async(self, query: str) -> str:
        """
        Use the AskNews `news` endpoint to get news context for your query. Remove irrelevant news. This code is mostly taken directly from the function of the same name in the parent class.
        """
        # AskNews has the harshest monthly API limits; if we want to run tests multiple times, it would be very good if we could reuse reports
        last_report: TimestampedAskNewsSearch | None = self._check_for_report()
        report = []
        if last_report is None or self._check_report_stale(last_report):
            async with AsyncAskNewsSDK(
                client_id=self.client_id,
                client_secret=self.client_secret,
                api_key=self.api_key,
                scopes=set(["news"]),
            ) as ask:
                await self.rate_limiter.acquire(1)
                hot_response = await ask.news.search_news(
                    query=query,  # your natural language query
                    n_articles=6,  # control the number of articles to include in the context, originally 5
                    return_type="both",
                    strategy="latest news",  # enforces looking at the latest news only
                )

                # get context from the "historical" database that contains a news archive going back to 2023
                await self.rate_limiter.acquire(5)
                historical_response = await ask.news.search_news(
                    query=query,
                    n_articles=10,
                    return_type="both",
                    strategy="news knowledge",  # looks for relevant news within the past 160 days
                )
                hot_articles = hot_response.as_dicts
                historical_articles = historical_response.as_dicts
                report = (hot_articles if hot_articles else []) + (
                    historical_articles if historical_articles else []
                )
                self._write_report(report)
        else:
            report = last_report.report

        relevant_articles = []
        for article in report:
            try:
                if await self.check_summary(query, article):
                    relevant_articles.append(article)
            except Exception as e:
                relevant_articles.append(article)

        formatted_articles = self._format_articles(relevant_articles)

        return formatted_articles

    def _check_for_report(self) -> TimestampedAskNewsSearch | None:
        if self.report_dir is not None:
            path = self._get_file_path()
            if os.path.exists(path):
                with open(path) as file:
                    report_json = json.load(file)
                    report_str = json.dumps(report_json)
                    report = TimestampedAskNewsSearch.model_validate_json(report_str)
                    return report
            else:
                return None
        else:
            return None

    def _write_report(self, report: List[SearchResponseDictItem]):
        if self.report_dir is not None and report:
            path = self._get_file_path()
            file_manipulation._create_directory_if_needed(path)
            with open(path, "w") as file:
                timestamped_report = TimestampedAskNewsSearch(
                    timestamp=time.time(), report=report
                )
                file.write(timestamped_report.model_dump_json())

    def _get_file_path(self) -> str:
        assert self.report_dir is not None, "Folder to save research to is not set"

        return os.path.join(
            self.report_dir, f"{self.question.id_of_question}_asknews_search.json"
        )

    def _check_report_stale(self, last_report: TimestampedAskNewsSearch) -> bool:
        report_time = datetime.fromtimestamp(last_report.timestamp)
        current_time = datetime.fromtimestamp(time.time())
        return (current_time - report_time).days >= self.freshness_threshold_days
