"""Shared wiring - one place that builds the registry, memory, router and agent."""
from __future__ import annotations

from app.assistant.agent import JarvusAgent
from app.assistant.router import Confirmer, ToolRouter
from app.audit.logger import AuditLogger
from app.config import Settings
from app.memory.enhanced import EnhancedMemory
from app.memory.store import MemoryStore
from app.models.base import ModelProvider
from app.models.local_provider import LocalProvider
from app.models.openai_provider import OpenAIProvider
from app.tools.clock import ClockTool
from app.tools.code import CreateCodeTool, PythonExecTool
from app.tools.docx import CreateDocxTool
from app.tools.filesystem import GlobTool, ListDirTool, ReadFileTool, WriteFileTool
from app.tools.internet import CryptoTool, CurrencyTool, JokeTool, NetworkInfoTool, NewsTool, StockTool, WikipediaTool
from app.tools.macos import OpenAppTool, SayTool
from app.tools.macos_control import BatteryTool, ClipboardTool, FrontAppTool, NotifyTool, ScreenshotTool, VolumeTool
from app.tools.macos_extra import DarkModeTool, EmptyTrashTool, FindFilesTool, LockScreenTool, OpenUrlTool, ShortcutTool, SleepDisplayTool
from app.tools.memory_tools import ForgetTool, RecallTool, RememberTool
from app.tools.messaging import EmailTool, MessageTool
from app.tools.music import MusicTool
from app.tools.notes import NoteTool
from app.tools.notifications import NotificationsTool
from app.tools.briefing import DailyBriefTool
from app.tools.devin import DevinProjectTool, DevinProjectsTool, DevinSendTool
from app.tools.plan import PlanTool, TodoWriteTool
from app.tools.registry import EchoTool, ToolRegistry
from app.tools.reminders import CalendarReadTool, CalendarTool, ListRemindersTool, ReminderTool
from app.tools.safe_shell import SafeShellTool
from app.tools.search import WebSearchTool
from app.tools.shell_extended import ShellReadOnlyTool
from app.tools.system import PsTool, SystemInfoTool
from app.tools.terminal import RunCommandTool
from app.tools.timer import AlarmTool, TimerTool
from app.tools.utilities import CalcTool, DateCalcTool, RandomTool, TextTool, UnitConvertTool
from app.tools.weather import WeatherTool
from app.tools.web import WebFetchTool


def build_registry(memory: EnhancedMemory) -> ToolRegistry:
    return ToolRegistry([
        EchoTool(),
        SafeShellTool(),
        ReadFileTool(),
        WriteFileTool(),
        ListDirTool(),
        GlobTool(),
        ShellReadOnlyTool(),
        SystemInfoTool(),
        PsTool(),
        ClockTool(),
        PlanTool(memory),
        TodoWriteTool(memory),
        PythonExecTool(),
        CreateCodeTool(),
        RunCommandTool(),
        OpenAppTool(),
        SayTool(),
        CreateDocxTool(),
        WeatherTool(),
        TimerTool(),
        AlarmTool(),
        ReminderTool(),
        CalendarTool(),
        ListRemindersTool(),
        CalendarReadTool(),
        DailyBriefTool(memory),
        MusicTool(),
        MessageTool(),
        EmailTool(),
        WebSearchTool(),
        WebFetchTool(),
        VolumeTool(),
        NotifyTool(),
        NotificationsTool(),
        ScreenshotTool(),
        ClipboardTool(),
        FrontAppTool(),
        BatteryTool(),
        NoteTool(),
        CalcTool(),
        UnitConvertTool(),
        DateCalcTool(),
        RandomTool(),
        TextTool(),
        CurrencyTool(),
        CryptoTool(),
        StockTool(),
        NewsTool(),
        WikipediaTool(),
        JokeTool(),
        NetworkInfoTool(),
        FindFilesTool(),
        OpenUrlTool(),
        DarkModeTool(),
        LockScreenTool(),
        SleepDisplayTool(),
        ShortcutTool(),
        EmptyTrashTool(),
        RememberTool(memory),
        RecallTool(memory),
        ForgetTool(memory),
        DevinProjectTool(memory),
        DevinProjectsTool(memory),
        DevinSendTool(memory),
    ])


def build_provider(
    settings: Settings,
    *,
    use_openai: bool = False,
    model: str | None = None,
    base_url: str | None = None,
) -> ModelProvider:
    if use_openai:
        api_key = settings.get_openai_api_key()
        return OpenAIProvider(api_key=api_key, model=model or settings.model)
    return LocalProvider(
        base_url=base_url or settings.llm_base_url,
        model=model or settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        num_ctx=settings.llm_num_ctx,
        think=settings.llm_think,
        keep_alive=settings.llm_keep_alive,
    )


def _seed_devin_projects(memory: EnhancedMemory, settings: Settings) -> None:
    """Merge JARVUS_DEVIN_PROJECTS entries into the stored registry (memory
    wins for names already registered so env edits don't clobber updates)."""
    seed = settings.devin_project_map
    if not seed:
        return
    from app.tools.devin import _load_registry, _save_registry
    reg = _load_registry(memory)
    changed = False
    for name, path in seed.items():
        if name not in reg:
            reg[name] = {"path": path, "cloud": False}
            changed = True
    if changed:
        _save_registry(memory, reg)


def build_agent(
    settings: Settings,
    *,
    use_openai: bool = False,
    model: str | None = None,
    base_url: str | None = None,
    confirmer: Confirmer | None = None,
) -> tuple[JarvusAgent, MemoryStore]:
    memory = EnhancedMemory(settings.resolved_db_path())
    _seed_devin_projects(memory, settings)
    registry = build_registry(memory)
    audit = AuditLogger(memory)
    router = ToolRouter(registry, audit, confirmer=confirmer)
    provider = build_provider(settings, use_openai=use_openai, model=model, base_url=base_url)
    agent = JarvusAgent(provider=provider, registry=registry, router=router, memory=memory,
                        tool_prefilter=settings.tool_prefilter,
                        auto_learn=settings.auto_learn)
    return agent, memory
