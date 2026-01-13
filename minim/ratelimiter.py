#!/usr/bin/env python3

import asyncio
from typing import Any
from aiolimiter import AsyncLimiter
from heapq import heappop, heappush
from functools import partial
from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.general_llm import (
    ModelInputType,
    TextTokenCostResponse,
)
from forecasting_tools.ai_models.model_interfaces.retryable_model import RetryableModel


class UnboundedAsyncLimiter(AsyncLimiter):
    """
    This rate limiter matches the behaviour of AsyncLimiter except that it ALWAYS allows acquisitions when the bucket is empty.
    """

    async def acquire(self, amount: float = 1) -> None:
        """Acquire capacity in the limiter.

        If the bucket is empty, returns immediately. Otherwise, if the limit has been reached, blocks until enough capacity has been freed before returning.

        :param amount: How much capacity you need to be available.
        :exception: Raises :exc:`ValueError` if `amount` is greater than
           :attr:`max_rate`.
        """

        loop = self._loop
        while not self.has_capacity(amount):
            # Add a future to the _waiters heapq to be notified when capacity
            # has come up. The future callback uses call_soon so other tasks
            # are checked *after* completing capacity acquisition in this task.
            fut = loop.create_future()
            fut.add_done_callback(partial(loop.call_soon, self._wake_next))
            heappush(self._waiters, (amount, self._next_count(), fut))
            self._wake_next()
            await fut

        self._level += amount
        # reset the waker to account for the new, lower level.
        self._wake_next()

        return None

    def has_capacity(self, amount: float = 1) -> bool:
        """Check if there is enough capacity remaining in the limiter or the limiter is at full capacity.

        :param amount: How much capacity you need to be available.

        """
        self._leak()
        return self._level == 0 or self._level + amount <= self.max_rate

    def _wake_next(self, *_args: object) -> None:
        """Wake the next waiting future or set a timer"""
        # clear timer and any cancelled futures at the top of the heap
        heap, handle, self._waker_handle = self._waiters, self._waker_handle, None
        if handle is not None:
            handle.cancel()
        while heap and heap[0][-1].done():
            heappop(heap)

        if not heap:
            # nothing left waiting
            return

        amount, _, fut = heap[0]
        self._leak()
        needed = min(amount - self.max_rate + self._level, self._level)
        if needed <= 0:
            heappop(heap)
            fut.set_result(None)
            # fut.set_result triggers another _wake_next call
            return

        wake_next_at = self._last_check + (1 / self._rate_per_sec * needed)
        self._waker_handle = self._loop.call_at(wake_next_at, self._wake_next)


class RateLimitedLlm(GeneralLlm):
    def __init__(
        self,
        model: str,
        rate_limiter: AsyncLimiter,
        responses_api: bool = False,
        allowed_tries: int = RetryableModel._DEFAULT_ALLOWED_TRIES,
        temperature: float | int | None = None,
        timeout: float | int | None = None,
        pass_through_unknown_kwargs: bool = True,
        populate_citations: bool = True,
        **kwargs,
    ) -> None:
        GeneralLlm.__init__(
            self,
            model,
            responses_api=responses_api,
            allowed_tries=allowed_tries,
            temperature=temperature,
            timeout=timeout,
            pass_through_unknown_kwargs=pass_through_unknown_kwargs,
            populate_citations=populate_citations,
            **kwargs,
        )
        self.rate_limiter = rate_limiter

    async def _mockable_direct_call_to_model(
        self, prompt: ModelInputType
    ) -> TextTokenCostResponse:
        await self.rate_limiter.acquire()
        return await super()._mockable_direct_call_to_model(prompt)
