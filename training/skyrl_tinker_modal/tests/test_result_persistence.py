"""CPU-only fault injection against patched SkyRL methods.

Set SKYRL_SOURCE_ROOT to a checkout with result_persistence.patch applied.
Only the methods under test are loaded, avoiding SkyRL's GPU dependencies.
"""
import ast
import asyncio
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlmodel")
pytest.importorskip("aiosqlite")
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError, TimeoutError as PoolTimeoutError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Field, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


class RequestStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class FutureDB(SQLModel, table=True):
    __tablename__ = "futures"
    request_id: int = Field(primary_key=True)
    result_data: str | None = None
    status: RequestStatus = RequestStatus.PENDING
    completed_at: datetime | None = None


class ErrorResponse(BaseModel):
    error: str
    status: str


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@pytest.fixture
def code():
    source = os.environ.get("SKYRL_SOURCE_ROOT")
    if not source:
        pytest.skip("Set SKYRL_SOURCE_ROOT to the patched pinned SkyRL checkout")
    root = Path(source)
    async def fast_sleep(delay):
        await asyncio.sleep(min(delay, 0.02))
    namespace = dict(
        asyncio=SimpleNamespace(sleep=fast_sleep, timeout=asyncio.timeout),
        datetime=datetime, timezone=timezone, AsyncSession=AsyncSession,
        FutureDB=FutureDB, RequestStatus=RequestStatus,
        OperationalError=OperationalError, PoolTimeoutError=PoolTimeoutError,
        types=SimpleNamespace(ErrorResponse=ErrorResponse),
        logger=logging.getLogger("persistence-test"), HTTPException=HTTPException,
        RETRIEVE_FUTURE_TIMEOUT_SECONDS=1, ValidationError=ValueError,
        raw_json_response=lambda payload: payload,
        RetrieveFutureRequest=object, Request=object,
    )
    tree = ast.parse((root / "skyrl/tinker/extra/skyrl_train_inference_forwarding.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.body = [n for n in cls.body if isinstance(n, ast.AsyncFunctionDef)
                and n.name in ("call_and_store_result", "_store_result_with_retry")]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "forwarding.py", "exec"), namespace)
    tree = ast.parse((root / "skyrl/tinker/api.py").read_text())
    names = {"finish_forwarded_request", "wait_for_future", "retrieve_future"}
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    for n in selected:
        n.decorator_list = []
        n.returns = None
        for arg in n.args.args:
            arg.annotation = None
    # The waiter uses the real asyncio module, including wait_for/get_running_loop.
    namespace["asyncio"] = asyncio
    exec(compile(ast.Module(body=selected, type_ignores=[]), "api.py", "exec"), namespace)
    # Keep the production retry sequence but shorten sleeps for fault injection.
    forwarding_globals = dict(namespace, asyncio=SimpleNamespace(sleep=fast_sleep, timeout=asyncio.timeout))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "forwarding.py", "exec"), forwarding_globals)
    namespace["client_class"] = forwarding_globals["SkyRLTrainInferenceForwardingClient"]
    return SimpleNamespace(**namespace)


async def database(tmp_path, n=1):
    path = tmp_path / "test.sqlite"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", connect_args={"timeout": 0.01})
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine) as session:
        session.add_all([FutureDB(request_id=i) for i in range(n)])
        await session.commit()
    blocker = sqlite3.connect(path)
    blocker.execute("PRAGMA journal_mode=WAL")
    return engine, blocker


def client(code, engine):
    obj = code.client_class()
    obj.db_engine = engine
    obj._result_write_lock = asyncio.Lock()
    return obj


def test_transient_sqlite_lock_preserves_all_results_without_regeneration(code, tmp_path):
    async def run():
        engine, blocker = await database(tmp_path, 32)
        obj = client(code, engine)
        generated = []
        async def generate(req, model_id, base_model=None):
            generated.append(req)
            return SimpleNamespace(model_dump_json=lambda: json.dumps({"tokens": [req]}))
        obj._forward_with_retry = generate
        blocker.execute("BEGIN IMMEDIATE")
        try:
            tasks = [asyncio.create_task(obj.call_and_store_result(i, i, "m", "v")) for i in range(32)]
            await asyncio.sleep(0.08)
            blocker.rollback()
            await asyncio.wait_for(asyncio.gather(*tasks), 3)
            async with AsyncSession(engine) as session:
                for i in range(32):
                    row = await session.get(FutureDB, i)
                    assert row.status == RequestStatus.COMPLETED
                    assert json.loads(row.result_data) == {"tokens": [i]}
            assert sorted(generated) == list(range(32))
        finally:
            blocker.close()
            await engine.dispose()
    asyncio.run(run())


def test_permanent_lock_reaches_existing_and_later_waiters(code, tmp_path):
    async def run():
        engine, blocker = await database(tmp_path)
        obj = client(code, engine)
        state = SimpleNamespace(future_waiters={}, forwarding_failures={})
        req = SimpleNamespace(app=SimpleNamespace(state=state))
        blocker.execute("BEGIN IMMEDIATE")
        waiter = asyncio.create_task(code.wait_for_future(state.future_waiters, 0, 1))
        task = asyncio.create_task(obj._store_result_with_retry(0, '{}', RequestStatus.COMPLETED))
        task.add_done_callback(lambda t: code.finish_forwarded_request(t, 0, state.future_waiters, state.forwarding_failures))
        try:
            with pytest.raises(OperationalError):
                await asyncio.wait_for(task, 2)
            status, data = await asyncio.wait_for(waiter, 1)
            assert status == RequestStatus.FAILED
            assert 'persist sampling result 0' in json.loads(data)['error']
            # Both later polling and HTTP retries see a terminal failure,
            # without consulting the still-locked database.
            for _ in range(2):
                with pytest.raises(HTTPException) as caught:
                    await code.retrieve_future(SimpleNamespace(request_id='0'), req)
                assert caught.value.status_code == 400
        finally:
            blocker.rollback()
            blocker.close()
            await engine.dispose()
    asyncio.run(run())


def test_pool_exhaustion_is_retried_with_fresh_sessions(code, tmp_path):
    async def run():
        engine, blocker = await database(tmp_path)
        obj = client(code, engine)
        globals_ = obj._store_result_with_retry.__globals__
        real_session = globals_["AsyncSession"]
        attempts = []
        def flaky_session(engine):
            attempts.append(1)
            if len(attempts) < 3:
                raise PoolTimeoutError('injected exhausted connection pool')
            return real_session(engine)
        globals_["AsyncSession"] = flaky_session
        try:
            await obj._store_result_with_retry(0, '{}', RequestStatus.COMPLETED)
            assert len(attempts) == 3
            async with real_session(engine) as session:
                assert (await session.get(FutureDB, 0)).status == RequestStatus.COMPLETED
        finally:
            blocker.close()
            await engine.dispose()
    asyncio.run(run())


def test_unrelated_db_errors_are_not_retried(code):
    async def run():
        obj = client(code, None)
        attempts = []
        def broken_session(engine):
            attempts.append(1)
            raise OperationalError('UPDATE', {}, sqlite3.OperationalError('no such table: futures'))
        obj._store_result_with_retry.__globals__["AsyncSession"] = broken_session
        with pytest.raises(OperationalError):
            await obj._store_result_with_retry(0, '{}', RequestStatus.COMPLETED)
        assert len(attempts) == 1
    asyncio.run(run())


def test_cancellation_does_not_publish_false_failure(code):
    async def run():
        async def pending():
            await asyncio.sleep(10)
        task = asyncio.create_task(pending())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        failures = {}
        code.finish_forwarded_request(task, 0, {}, failures)
        assert not failures
    asyncio.run(run())


def test_missing_row_reports_failure_instead_of_silent_success(code, tmp_path):
    async def run():
        engine, blocker = await database(tmp_path, 0)
        try:
            with pytest.raises(RuntimeError, match='missing'):
                await client(code, engine)._store_result_with_retry(123, '{}', RequestStatus.COMPLETED)
        finally:
            blocker.close()
            await engine.dispose()
    asyncio.run(run())


def test_persistence_deadline_includes_waiting_for_writer_lock(code):
    async def run():
        obj = client(code, None)
        async def generate(*args, **kwargs):
            return SimpleNamespace(model_dump_json=lambda: '{}')
        obj._forward_with_retry = generate
        budgets = []
        def short_timeout(seconds):
            budgets.append(seconds)
            return asyncio.timeout(0.02)
        obj.call_and_store_result.__globals__["asyncio"].timeout = short_timeout
        await obj._result_write_lock.acquire()
        task = asyncio.create_task(obj.call_and_store_result(0, None, 'm', 'v'))
        failures = {}
        task.add_done_callback(lambda t: code.finish_forwarded_request(t, 0, {}, failures))
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, 1)
            assert budgets == [180]
            assert failures[0][0] == RequestStatus.FAILED
            assert 'TimeoutError' in failures[0][1]
            assert obj._result_write_lock.locked()
        finally:
            obj._result_write_lock.release()
    asyncio.run(run())


def test_generation_error_is_persisted_as_terminal_failure(code, tmp_path):
    async def run():
        engine, blocker = await database(tmp_path)
        obj = client(code, engine)
        async def generate(*args, **kwargs):
            raise ValueError('injected generation error')
        obj._forward_with_retry = generate
        try:
            await obj.call_and_store_result(0, None, 'm', 'v')
            async with AsyncSession(engine) as session:
                row = await session.get(FutureDB, 0)
                assert row.status == RequestStatus.FAILED
                assert json.loads(row.result_data)['error'] == 'injected generation error'
        finally:
            blocker.close()
            await engine.dispose()
    asyncio.run(run())
