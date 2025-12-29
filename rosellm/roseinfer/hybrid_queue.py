from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Deque, Generic, TypeVar

T = TypeVar("T")


class HybridQueue(Generic[T]):
    """A tiny sync+async queue for a single event loop.

    - Producers can call `put()` from any thread.
    - Consumers can either block with `get()` (threads) or await `aget()` (async).

    This avoids holding a worker thread per streaming HTTP connection when used
    with an async server.
    """

    __slots__ = ("_buf", "_cv", "_loop", "_event")

    def __init__(self) -> None:
        self._buf: Deque[T] = deque()
        self._cv = threading.Condition()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None

    def set_asyncio_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._cv:
            self._loop = loop
            self._event = asyncio.Event()
            if self._buf:
                self._event.set()

    def put(self, item: T) -> None:
        with self._cv:
            self._buf.append(item)
            self._cv.notify()
            loop = self._loop
            event = self._event
        if loop is not None and event is not None:
            loop.call_soon_threadsafe(event.set)

    def get(self) -> T:
        with self._cv:
            while not self._buf:
                self._cv.wait()
            return self._buf.popleft()

    async def aget(self) -> T:
        while True:
            with self._cv:
                if self._buf:
                    return self._buf.popleft()
                event = self._event
                if event is not None:
                    event.clear()
            if event is None:
                return await asyncio.to_thread(self.get)
            await event.wait()
