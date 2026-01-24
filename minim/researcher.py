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

from pydantic import BaseModel, ValidationError

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
    query: str
    report: List[SearchResponseDictItem]


class MinimResearcher:
    """
    This is the researcher for the minim forecasting bot. Currently, it follows the following procedure to produce research:
    1. (Potentially) check whether the question text is an acceptable query for the AskNews API.
    2. If not, produce better queries and search with those; otherwise, search with the question text.
    3. (Potentially) remove irrelevant articles.
    4. Return all relevant articles.
    """

    _asknews_rate_limit = 12.0

    def __init__(
        self,
        parser: GeneralLlm,
        general_model: GeneralLlm,
        asknews_researcher: str,
        report_dir: str | None = None,
        check_query: bool = False,
        check_relevance: bool = True,
    ):
        self.parser = parser
        self.general_model = general_model
        self.asknews_researcher = asknews_researcher
        self.asknews_limiter = UnboundedAsyncLimiter(1, self._asknews_rate_limit)
        self.report_dir = report_dir
        self.check_query = check_query
        self.check_relevance = check_relevance

    async def run_research(self, question: MetaculusQuestion) -> str:
        research = ""

        asknewsquery = "v0.1"  # this is a required argument to the searcher, so it's been repurposed as a caching check
        asknewsresearch = await MinimAskNewsSearcher(
            parser=self.parser,
            general_model=self.general_model,
            question=question,
            rate_limiter=self.asknews_limiter,
            report_dir=self.report_dir,
            check_query=self.check_query,
            check_relevance=self.check_relevance,
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
        general_model: GeneralLlm,
        question: MetaculusQuestion,
        rate_limiter: UnboundedAsyncLimiter,
        report_dir: str | None = None,
        check_query: bool,
        check_relevance: bool,
    ):
        AskNewsSearcher.__init__(self)
        self.parser = parser
        self.general_model = general_model
        self.question = question
        self.rate_limiter = rate_limiter
        self.report_dir = report_dir
        self.check_query = check_query
        self.check_relevance = check_relevance

    async def do_check_query(self) -> bool:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster gives you a question they intend to forecast on, along with the criteria that determine how it will be decided, and will be sending a query to a news search tool. You decide whether the text of the question is an acceptable search query which will get search results relevant to the outcome of the question the superforecaster is forecasting.
            Most questions that the superforecaster gets are appropriate to use as search queries. The cases where they are not appropriate search queries are generally those with pronoun phrases which correspond to things not specified in the question text. For instance, the question "How many of the following US senators will be reelected?" is not an acceptable query, since the "the following US senators" refers to information needed for searching which is not in the question text. The question "Will Rand Paul be reelected senator?" is an acceptable query, even though the question itself is not totally sufficient to determine a prediction. Generally, you should err on the side of accepting the question text as a search query, and only decide that it is not acceptable if there is a serious absence which will make a search tool unable to find relevant results.
            The question the superforecaster gives you, and the text of the query you are considering, is:
            {self.question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {self.question.resolution_criteria}
            
            {self.question.fine_print}

            You need to decide whether the question text is an acceptable search query to find the news relevant to deciding the question.
            You first write your reasoning, explaining why the search query will produce the relevant news or not.
            Then, to finish your response, you write a line consisting of "Final answer: " followed by the word "true" or the word "false". You write "true" when the search query is acceptable, and "false" when it is not.
            """
        )

        response = await self.general_model.invoke(prompt)
        query_acceptable: bool = await structure_output(
            response,
            bool,
            model=self.parser,
            num_validation_samples=2,
        )
        logger.info(
            f'The text of the question "{self.question.question_text}" deemed '
            + "acceptable"
            if query_acceptable
            else "not acceptable" + " as a search query."
        )

        return query_acceptable

    async def produce_queries(self) -> tuple[str, list[str]]:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster gives you a question they intend to forecast on, along with the criteria that determine how it will be decided, and will be sending a query to a news search tool. They have determined that the text of the question is unacceptable as a search query to get relevant news results, since not all the information necessary for searching is present in the question text. You produce a query or list of queries that will be sent to the news search tool.
            The search tool has two modes: "historical", which searches for articles that are possibly years old, and "recent", which only searches for articles that have been published in the last few days. "Historical" searches are expensive, and so your team can only afford to make one "historical" query. You will thus first produce a single query which will be used for a "historical" search, and then produce possibly up to twenty additional queries for "recent" searches, since they are cheaper.
            Your queries should be in the form of questions about the future, like the text of the question to forecast.
            It will often not be possible to produce a single query which contains all the relevant information. For instance, if the question was "How many of the following US senators will be reelected?", with a resolution criteria that included fifteen US senators, would not be possible to distill into a single query. In this situation, you would do your best to produce a general question, without trying to include specific names. In this example, an acceptable query might be "Which US senators are unlikely to be reelected?". Note that you should not produce queries about this example question.
            After you have produced your "historical" query, you may augment it with up to twenty additional "recent" queries. These are chosen to each search for exactly one aspect of the full question. For instance, if the question was "How many of these oil refineries will be attacked by Russia?" and there was a list of three oil refineries, you might produce three "recent" queries each of the form "Will [refinery name] be attacked by Russia?". Again, you do not produce queries about this example scenario. You do not need to produce any "recent" queries at all, if your single "historical" query contains the relevant information for the search tool to use.
            
            The question the superforecaster gives you is:
            {self.question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {self.question.resolution_criteria}
            
            {self.question.fine_print}

            You need to produce one or more search queries to be sent to the news search tool.
            You first write your reasoning for selecting your "historical" query.
            Then, you write your reasoning for your selection of "recent" queries, if any. You produce from zero to twenty "recent" queries.
            Then, to finish your response, you write a line consisting only of "Final answer:", then a line consisting only of your "historical" query. Then, with each query on its own new line, you write each of your "recent" queries.
            """
        )

        response = await self.general_model.invoke(prompt)
        lines = iter(response.split("\n"))
        historical_query = ""
        recent_queries = []
        try:
            while "Final answer:" not in next(lines):
                pass
            historical_query = next(lines)
            logger.info(f'Historical search query: "{historical_query}"')
        except StopIteration as e:
            logger.error(
                f'Search query for question "{self.question.question_text}" not produced! Using question text as fallback query.'
            )
            historical_query = self.question.question_text
        recent_queries = [line for line in lines]
        if recent_queries:
            logger.info(f"Recent search queries:\n" + "\n".join(recent_queries))
        return (historical_query, recent_queries[:20])

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

        response = await self.general_model.invoke(prompt)
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
        if last_report is None or self._check_report_stale(query, last_report):
            async with AsyncAskNewsSDK(
                client_id=self.client_id,
                client_secret=self.client_secret,
                api_key=self.api_key,
                scopes=set(["news"]),
            ) as ask:
                historical_query = self.question.question_text
                hot_queries = []
                question_text_acceptable = True
                if self.check_query:
                    question_text_acceptable = await self.do_check_query()
                # NOTE: the template bot includes a full prompt. Past template bots, which have done well, have not; motivated by a desire not to fix what isn't broken, I've reverted back to including just the question. THIS CAUSES PROBLEMS, but they have empirically not been enough to make the bot not work. The acceptability check and query reconstruction is an attempt to fix some of the problems.

                if not question_text_acceptable:
                    (historical_query, hot_queries) = await self.produce_queries()

                await self.rate_limiter.acquire(1)
                hist_hot_response = await ask.news.search_news(
                    query=query,  # your natural language query
                    n_articles=6,  # control the number of articles to include in the context, originally 5
                    return_type="both",
                    strategy="latest news",  # enforces looking at the latest news only
                ).as_dicts
                # get context from the "historical" database that contains a news archive going back to 2023
                await self.rate_limiter.acquire(5)
                hist_full_response = await ask.news.search_news(
                    query=query,
                    n_articles=10,
                    return_type="both",
                    strategy="news knowledge",  # looks for relevant news within the past 160 days
                ).as_dicts
                hot_articles = hist_hot_response if hist_full_response else []
                historical_articles = hist_full_response if hist_full_response else []
                for query in hot_queries:
                    await self.rate_limiter.acquire(1)
                    hot_response = await ask.news.search_news(
                        query=query,
                        n_articles=5,
                        return_type="both",
                        strategy="latest news",
                    ).as_dicts
                    hot_articles.extend(hot_response if hot_response else [])
                report = hot_articles + historical_articles
                self._write_report(query, report)
        else:
            logger.info(
                f"Fresh AskNews report for question with ID {self.question.id_of_question} found and loaded."
            )
            report = last_report.report

        relevant_articles = []
        if self.check_relevance:
            for article in report:
                try:
                    if await self.check_summary(query, article):
                        relevant_articles.append(article)
                except Exception as e:
                    relevant_articles.append(article)
        else:
            relevant_articles = report

        formatted_articles = self._format_articles(relevant_articles)

        return formatted_articles

    def _check_for_report(self) -> TimestampedAskNewsSearch | None:
        if self.report_dir is not None:
            path = self._get_file_path()
            if os.path.exists(path):
                with open(path) as file:
                    report_json = json.load(file)
                    report_str = json.dumps(report_json)
                    try:
                        report = TimestampedAskNewsSearch.model_validate_json(
                            report_str
                        )
                        return report
                    except ValidationError as e:
                        logger.warning(
                            f"Research report found for question {self.question.id_of_question} but failed to validate."
                        )
                        return None
            else:
                return None
        else:
            return None

    def _write_report(self, query: str, report: List[SearchResponseDictItem]):
        if self.report_dir is not None and report:
            path = self._get_file_path()
            file_manipulation._create_directory_if_needed(path)
            with open(path, "w") as file:
                timestamped_report = TimestampedAskNewsSearch(
                    timestamp=time.time(), query=query, report=report
                )
                file.write(timestamped_report.model_dump_json())

    def _get_file_path(self) -> str:
        assert self.report_dir is not None, "Folder to save research to is not set"

        return os.path.join(
            self.report_dir, f"{self.question.id_of_question}_asknews_search.json"
        )

    def _check_report_stale(
        self, query: str, last_report: TimestampedAskNewsSearch
    ) -> bool:
        report_time = datetime.fromtimestamp(last_report.timestamp)
        current_time = datetime.fromtimestamp(time.time())
        old = (current_time - report_time).days >= self.freshness_threshold_days
        obsolete = query != last_report.query
        if old:
            logger.info(
                f"Old research report found for question {self.question.id_of_question}."
            )
        if obsolete:
            logger.info(
                f"Research report with obsolete query found for question {self.question.id_of_question}."
            )
        return old or obsolete
