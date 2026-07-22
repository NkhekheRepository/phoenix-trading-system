import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class Event:
    source: str
    event_type: str
    severity: str
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RateLimiter:
    def __init__(self):
        self._last_sent: Dict[str, datetime] = {}

    def can_send(self, key: str, min_interval_seconds: int = 300) -> bool:
        last = self._last_sent.get(key)
        if not last:
            self._last_sent[key] = datetime.now(timezone.utc)
            return True
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        if elapsed >= min_interval_seconds:
            self._last_sent[key] = datetime.now(timezone.utc)
            return True
        return False

    def reset(self, key: str):
        self._last_sent.pop(key, None)


class MessageFormatter:
    SEP = "━" * 30

    @staticmethod
    def regime_alert(old_regime: str, new_regime: str, confidence: float, mode: str, signals: Dict) -> str:
        tag = "[BULL]" if "BULL" in new_regime else "[BEAR]" if "BEAR" in new_regime else "[SIDEWAYS]"
        lines = [
            f"━━━ REGIME SHIFT ━━━",
            f"{old_regime} -> {new_regime} {tag}",
            f"Confidence: {confidence:.0%}",
            f"Mode: {mode}",
            f"Signals: Trend {signals.get('trend', 0):+.2f} | Mom {signals.get('momentum', 0):+.2f} | HMM {signals.get('hmm', 0):.2f}",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def regime_summary(regime: str, confidence: float, mode: str, persistence: int) -> str:
        lines = [
            f"━━━ 📊 REGIME STATUS ━━━",
            f"Regime: *{regime}*",
            f"Confidence: {confidence:.0%}",
            f"Mode: {mode}",
            f"Expected persistence: ~{persistence} periods",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def risk_alert(old_level: str, new_level: str, triggers: List[str], impact: Dict) -> str:
        emoji = "🚨" if new_level in ("PAUSED", "EMERGENCY") else "⚠️"
        lines = [
            f"━━━ {emoji} RISK ESCALATION ━━━",
            f"Level: {old_level} → *{new_level}*",
        ]
        if triggers:
            lines.append("Triggers:")
            for t in triggers:
                lines.append(f"  - {t}")
        if impact:
            lines.append("Impact:")
            for k, v in impact.items():
                lines.append(f"  - {k}: {v}")
        lines.append(MessageFormatter.SEP)
        return "\n".join(lines)

    @staticmethod
    def risk_update(level: str, triggers: List[str]) -> str:
        emoji = "✅" if level == "NORMAL" else "⚠️"
        lines = [
            f"━━━ {emoji} RISK STATUS ━━━",
            f"Level: *{level}*",
        ]
        if triggers:
            lines.append("Active warnings:")
            for t in triggers:
                lines.append(f"  - {t}")
        lines.append(MessageFormatter.SEP)
        return "\n".join(lines)

    @staticmethod
    def daily_risk(drawdown: float, exposure: float, leverage: float, consecutive: int, level: str) -> str:
        lines = [
            f"━━━ 📋 RISK REPORT ━━━",
            f"Level: *{level}*",
            f"Daily DD: {drawdown:.1%}",
            f"Exposure: {exposure:.1%}",
            f"Leverage: {leverage:.0f}x",
            f"Consecutive losses: {consecutive}",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def trade_attribution(pair: str, side: str, profit_pct: float, exit_reason: str,
                           entry_tag: str, regime: str, failure_factors: List[str],
                           success_factors: List[str], duration: str) -> str:
        is_win = profit_pct >= 0
        emoji = "✅" if is_win else "❌"
        result = f"+{profit_pct:.2f}%" if is_win else f"{profit_pct:.2f}%"
        lines = [
            f"━━━ {emoji} TRADE ATTRIBUTION ━━━",
            f"*{pair}* | {side} | {result}",
            f"Duration: {duration} | Exit: {exit_reason}",
            f"Entry: {entry_tag}",
            f"Regime: {regime}",
        ]
        if failure_factors:
            lines.append("Failure factors:")
            for f in failure_factors:
                lines.append(f"  - {f}")
        if success_factors:
            lines.append("Success factors:")
            for f in success_factors:
                lines.append(f"  - {f}")
        lines.append(MessageFormatter.SEP)
        return "\n".join(lines)

    @staticmethod
    def drift_alert(feature: str, psi: float, kl: float, wasserstein: float,
                    severity: str, recommendation: str) -> str:
        lines = [
            f"━━━ 🔬 CONCEPT DRIFT ━━━",
            f"Feature: *{feature}*",
            f"PSI: {psi:.4f} | KL: {kl:.4f} | WD: {wasserstein:.4f}",
            f"Severity: *{severity.upper()}*",
            f"Recommendation: {recommendation}",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def experiment_result(exp_id: str, hypothesis: str, results: Dict,
                           decision: str, reason: str) -> str:
        emoji = "✅" if decision == "approved" else "❌" if decision == "rejected" else "⏳"
        lines = [
            f"━━━ 🧪 EXPERIMENT ━━━",
            f"ID: `{exp_id}`",
            f"Hypothesis: {hypothesis}",
        ]
        if results:
            lines.append("Results:")
            for k, v in results.items():
                lines.append(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")
        lines.append(f"Decision: {emoji} *{decision.upper()}*")
        lines.append(f"Reason: {reason}")
        lines.append(MessageFormatter.SEP)
        return "\n".join(lines)

    @staticmethod
    def allocation_change(allocations: List[Dict], cash_reserve: float, regime: str, risk: str) -> str:
        lines = [
            f"━━━ 🎯 STRATEGY ALLOCATION ━━━",
            f"Regime: {regime} | Risk: {risk}",
        ]
        for a in allocations:
            bar = "█" * int(a["weight"] * 20)
            lines.append(f"  {a['strategy']:20s} {bar} {a['weight']:.0%}")
        if cash_reserve > 0:
            lines.append(f"  {'CASH':20s} {'░' * int(cash_reserve * 20)} {cash_reserve:.0%}")
        lines.append(MessageFormatter.SEP)
        return "\n".join(lines)

    @staticmethod
    def memory_match(topic: str, insight: str, confidence: float) -> str:
        lines = [
            f"━━━ 🧠 MARKET MEMORY ━━━",
            f"Topic: *{topic}*",
            f"Insight: {insight}",
            f"Confidence: {confidence:.0%}",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def deployment_event(event_type: str, version: str, status: str, changelog: str = "") -> str:
        emoji = "🚀" if status in ("live", "canary") else "🔙" if status == "rolled_back" else "📦"
        lines = [
            f"━━━ {emoji} DEPLOYMENT ━━━",
            f"Event: *{event_type}*",
            f"Version: `{version}`",
            f"Status: {status}",
        ]
        if changelog:
            lines.append(f"Changes: {changelog[:200]}")
        lines.append(MessageFormatter.SEP)
        return "\n".join(lines)

    @staticmethod
    def research_hypothesis(hypothesis: Dict) -> str:
        lines = [
            f"━━━ 💡 RESEARCH HYPOTHESIS ━━━",
            f"Observation: {hypothesis.get('observation', '')}",
            f"Hypothesis: *{hypothesis.get('hypothesis', '')}*",
            f"Experiment: {hypothesis.get('suggested_experiment', '')}",
            f"Priority: {hypothesis.get('priority', 'medium')}",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def daily_summary(data: Dict) -> str:
        lines = [
            f"━━━ 📆 DAILY REPORT ━━━",
            f"📅 {data.get('date', '')} | *{data.get('bot_name', '')}*",
            "",
            "📊 Performance",
            f"  Trades: {data.get('trades', 0)} ({data.get('wins', 0)}W ✅ / {data.get('losses', 0)}L ❌)",
            f"  Win Rate: {data.get('win_rate', 0):.1%}",
            f"  P&L: {data.get('pnl', '')}",
        ]
        if data.get('best_trade'):
            lines.append(f"  Best: {data['best_trade']}")
        if data.get('worst_trade'):
            lines.append(f"  Worst: {data['worst_trade']}")
        lines.extend([
            "",
            f"🌦️ Regime: {data.get('regime', 'N/A')}",
            f"⚠️ Risk: {data.get('risk_level', 'N/A')}",
            f"📈 Exposure: {data.get('exposure', 0):.0%} | Leverage: {data.get('leverage', 0):.0f}x",
        ])
        if data.get('active_trades'):
            lines.append("")
            lines.append(f"Active Trades ({len(data['active_trades'])}/{data.get('max_trades', 5)})")
            for t in data['active_trades'][:5]:
                lines.append(f"  {t['pair']:12s} @ {t.get('rate', 0):,.0f}  {t.get('profit', ''):>8s}")
        lines.append(MessageFormatter.SEP)
        return "\n".join(lines)

    @staticmethod
    def bot_health(data: Dict) -> str:
        lines = [
            f"━━━ ❤️ BOT HEALTH ━━━",
            f"Bot: *{data.get('bot_name', '')}*",
            f"Uptime: {data.get('uptime', 'N/A')}",
            f"Trades total: {data.get('total_trades', 0)}",
            f"Active trades: {data.get('active_trades', 0)}",
            f"Memory: {data.get('memory_mb', 0):.0f} MB",
            f"Exchange: {data.get('exchange', 'binance')} ({'✅' if data.get('exchange_ok') else '❌'})",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def retrain_alert(triggers: List[Dict]) -> str:
        lines = [
            "━━━ 🤖 ML RETRAIN TRIGGERED ━━━",
        ]
        for t in triggers[:5]:
            lines.append(f"  - {t.get('source', '?')}: {t.get('metric', '?')}={t.get('value', '?'):.2f}")
        lines.append(MessageFormatter.SEP)
        return "\n".join(lines)

    @staticmethod
    def retrain_scheduled(exp_id: str, trigger: str) -> str:
        lines = [
            "━━━ 🧪 RETRAIN SCHEDULED ━━━",
            f"Experiment: `{exp_id}`",
            f"Trigger: {trigger}",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def validation_gate_report(version: str, passed: bool, gates: str) -> str:
        emoji = "✅" if passed else "❌"
        lines = [
            f"━━━ 🔬 VALIDATION REPORT ━━━",
            f"Version: `{version}`",
            f"Result: {emoji} *{'PASSED' if passed else 'FAILED'}* ({gates})",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def promotion_decision(champion: str, challenger: str, promote: bool, reasons: List[str]) -> str:
        emoji = "🏆" if promote else "📊"
        lines = [
            f"━━━ {emoji} CHAMPION CHALLENGER ━━━",
            f"Champion: {champion}",
            f"Challenger: {challenger}",
            f"Decision: {'✅ PROMOTE' if promote else '❌ REJECT'}",
        ]
        if reasons:
            lines.append("Reasons:")
            for r in reasons:
                lines.append(f"  - {r}")
        lines.append(MessageFormatter.SEP)
        return "\n".join(lines)

    @staticmethod
    def rollback_alert(version: str, reason: str) -> str:
        lines = [
            "━━━ 🔴 AUTO ROLLBACK ━━━",
            f"Rolled back to: `{version}`",
            f"Reason: {reason}",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def market_event(event_type: str, description: str, impact: str) -> str:
        lines = [
            "━━━ 📌 MARKET EVENT ━━━",
            f"Type: {event_type}",
            f"Description: {description}",
            f"Impact: {impact}",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def knowledge_added(topic: str, insight: str, source: str) -> str:
        lines = [
            "━━━ 📚 KNOWLEDGE ADDED ━━━",
            f"Topic: *{topic}*",
            f"Insight: {insight[:200]}",
            f"Source: {source}",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def data_quality_alert(pair: str, issue: str, details: str) -> str:
        lines = [
            "━━━ ⚠️ DATA QUALITY ━━━",
            f"Pair: {pair}",
            f"Issue: {issue}",
            f"Details: {details}",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def strategy_note(strategy: str, note: str) -> str:
        lines = [
            "━━━ 📝 STRATEGY NOTE ━━━",
            f"Strategy: {strategy}",
            f"Note: {note[:200]}",
            MessageFormatter.SEP,
        ]
        return "\n".join(lines)


class Monitor:
    def __init__(self, dp=None, bot_name: str = "PhoenixScalper", chat_id: str = None, token: str = None):
        self.dp = dp
        self.bot_name = bot_name
        self.chat_id = chat_id
        self.token = token
        self._queue: deque[Event] = deque(maxlen=200)
        self._rate_limiter = RateLimiter()
        self._last_regime: Optional[str] = None
        self._last_risk_level: Optional[str] = None
        self._last_daily_date: Optional[str] = None
        self._last_hourly_sent: Optional[datetime] = None
        self._trade_count_since_report = 0
        self._wins_since_report = 0
        self._losses_since_report = 0
        self._daily_pnl: float = 0.0
        self._best_trade: Optional[str] = None
        self._worst_trade: Optional[str] = None
        self._best_profit: float = -999.0
        self._worst_profit: float = 999.0

    def on_event(self, event: Event):
        self._queue.append(event)
        if event.severity in ("critical", "emergency"):
            self._send_now(event)

    def _send_now(self, event: Event):
        if self.dp and hasattr(self.dp, 'send_msg'):
            try:
                self.dp.send_msg(event.message, always_send=True)
            except Exception as e:
                logger.warning(f"Monitor send failed: {e}")
        elif self.token and self.chat_id:
            self._send_telegram_direct(event.message)

    def _send_telegram_direct(self, text: str):
        import urllib.request
        import urllib.parse
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }).encode()
            urllib.request.urlopen(url, data=data, timeout=10)
        except Exception as e:
            logger.warning(f"Direct Telegram send failed: {e}")

    def notify_regime_change(self, old: str, new: str, confidence: float, mode: str, signals: Dict):
        rate_key = f"regime_{new}"
        if not self._rate_limiter.can_send(rate_key, 1800):
            return
        msg = MessageFormatter.regime_alert(old, new, confidence, mode, signals)
        self._queue.append(Event("regime_engine", "regime_change", "info", "Regime change", msg))

    def notify_risk_change(self, level: str, triggers: List[str], impact: Dict):
        rate_key = f"risk_{level}"
        min_int = 300 if level in ("CAUTION", "REDUCED") else 60
        if not self._rate_limiter.can_send(rate_key, min_int):
            return
        old = self._last_risk_level or "UNKNOWN"
        if level in ("PAUSED", "EMERGENCY"):
            msg = MessageFormatter.risk_alert(old, level, triggers, impact)
            self._queue.append(Event("risk_governor", "risk_alert", "critical", "Risk escalation", msg))
        elif level == "NORMAL" and old != "NORMAL":
            msg = MessageFormatter.risk_update(level, triggers)
            self._queue.append(Event("risk_governor", "risk_update", "info", "Risk normalized", msg))
        elif level != old:
            msg = MessageFormatter.risk_alert(old, level, triggers, impact)
            self._queue.append(Event("risk_governor", "risk_change", "warning", "Risk change", msg))
        self._last_risk_level = level

    def notify_drift(self, feature: str, psi: float, kl: float, wd: float, severity: str, recommendation: str):
        rate_key = f"drift_{feature}"
        if not self._rate_limiter.can_send(rate_key, 3600):
            return
        msg = MessageFormatter.drift_alert(feature, psi, kl, wd, severity, recommendation)
        self._queue.append(Event("concept_drift", "drift", "warning" if severity == "warning" else "info", "Drift detected", msg))

    def notify_experiment(self, exp_id: str, hypothesis: str, results: Dict, decision: str, reason: str):
        msg = MessageFormatter.experiment_result(exp_id, hypothesis, results, decision, reason)
        sev = "info" if decision == "approved" else "warning"
        self._queue.append(Event("experiment_db", "experiment", sev, "Experiment result", msg))

    def notify_allocation(self, allocations: List[Dict], cash_reserve: float, regime: str, risk: str):
        rate_key = "allocation"
        if not self._rate_limiter.can_send(rate_key, 1800):
            return
        msg = MessageFormatter.allocation_change(allocations, cash_reserve, regime, risk)
        self._queue.append(Event("strategy_allocator", "allocation", "info", "Allocation change", msg))

    def notify_memory_match(self, topic: str, insight: str, confidence: float):
        msg = MessageFormatter.memory_match(topic, insight, confidence)
        self._queue.append(Event("market_memory", "memory_match", "info", "Memory match", msg))

    def notify_deployment(self, event_type: str, version: str, status: str, changelog: str = ""):
        msg = MessageFormatter.deployment_event(event_type, version, status, changelog)
        sev = "warning" if status == "rolled_back" else "info"
        self._queue.append(Event("deployment", event_type, sev, "Deployment", msg))

    def notify_research(self, hypothesis: Dict):
        msg = MessageFormatter.research_hypothesis(hypothesis)
        self._queue.append(Event("research", "hypothesis", "info", "Research hypothesis", msg))

    def notify_retrain_triggers(self, triggers: List[Dict]):
        rate_key = "retrain_triggers"
        if not self._rate_limiter.can_send(rate_key, 3600):
            return
        msg = MessageFormatter.retrain_alert(triggers)
        self._queue.append(Event("ml_engine", "retrain_triggers", "info", "ML retrain triggered", msg))

    def notify_retrain_scheduled(self, experiment_id: str, trigger: str):
        msg = MessageFormatter.retrain_scheduled(experiment_id, trigger)
        self._queue.append(Event("ml_engine", "retrain_scheduled", "info", "Retrain scheduled", msg))

    def notify_validation_complete(self, version: str, passed: bool, gates: str, results: Dict = None):
        msg = MessageFormatter.validation_gate_report(version, passed, gates)
        self._queue.append(Event("validation_pipeline", "validation_complete",
                                 "info" if passed else "warning",
                                 "Validation complete", msg))

    def notify_champion_challenger_eval(self, champion: str, challenger: str,
                                         promote: bool, reasons: List[str] = None):
        rate_key = "cc_eval"
        if not self._rate_limiter.can_send(rate_key, 3600):
            return
        msg = MessageFormatter.promotion_decision(champion, challenger, promote, reasons)
        sev = "info" if promote else "debug"
        self._queue.append(Event("champion_challenger", "champion_challenger_eval",
                                 sev, "Champion/challenger eval", msg))

    def notify_champion_promoted(self, version: str, previous: str, reasons: List[str]):
        msg = MessageFormatter.promotion_decision(previous, version, True, reasons)
        self._queue.append(Event("champion_challenger", "champion_promoted",
                                 "info", "Champion promoted", msg))

    def notify_auto_rollback(self, version: str, sharpe: float, threshold: float):
        reason = f"Live Sharpe {sharpe:.2f} < champion baseline {threshold:.2f}"
        msg = MessageFormatter.rollback_alert(version, reason)
        self._queue.append(Event("champion_challenger", "auto_rollback",
                                 "warning", "Auto rollback", msg))

    def notify_market_event(self, event_type: str, description: str, impact: str):
        rate_key = f"market_{event_type}"
        if not self._rate_limiter.can_send(rate_key, 3600):
            return
        msg = MessageFormatter.market_event(event_type, description, impact)
        self._queue.append(Event("market_memory", "market_event", "info", "Market event", msg))

    def notify_knowledge_added(self, topic: str, insight: str, source: str):
        msg = MessageFormatter.knowledge_added(topic, insight, source)
        self._queue.append(Event("market_memory", "knowledge_added", "info", "Knowledge added", msg))

    def notify_data_quality(self, pair: str, issue: str, details: str):
        rate_key = f"dq_{pair}_{issue}"
        if not self._rate_limiter.can_send(rate_key, 600):
            return
        msg = MessageFormatter.data_quality_alert(pair, issue, details)
        sev = "warning" if issue in ("stale_data", "missing_candles", "nan_values") else "info"
        self._queue.append(Event("data_quality", "data_quality", sev, "Data quality", msg))

    def notify_strategy_note(self, strategy: str, note: str):
        msg = MessageFormatter.strategy_note(strategy, note)
        self._queue.append(Event("market_memory", "strategy_note", "info", "Strategy note", msg))

    def notify_trade_attribution(self, pair: str, side: str, profit_pct: float,
                                  exit_reason: str, entry_tag: str, regime: str,
                                  failure_factors: List[str], success_factors: List[str],
                                  duration: str):
        msg = MessageFormatter.trade_attribution(
            pair, side, profit_pct, exit_reason, entry_tag, regime,
            failure_factors, success_factors, duration
        )
        self._queue.append(Event("trade_intel", "trade_attribution", "info", "Trade attribution", msg))

        if profit_pct >= 0:
            self._wins_since_report += 1
            if not self._best_trade or profit_pct > self._best_profit:
                self._best_trade = f"{pair} {side} +{profit_pct:.2f}%"
                self._best_profit = profit_pct
        else:
            self._losses_since_report += 1
            if not self._worst_trade or profit_pct < self._worst_profit:
                self._worst_trade = f"{pair} {side} {profit_pct:.2f}%"
                self._worst_profit = profit_pct
        self._trade_count_since_report += 1
        self._daily_pnl += profit_pct / 100

    def send_daily_summary(self, date_str: str, regime: str, risk_level: str,
                            exposure: float, leverage: float, active_trades: List[Dict],
                            max_trades: int, bot_name: str = None, total_trades_db: int = 0,
                            total_profit: float = 0.0, win_count: int = 0, loss_count: int = 0):
        if self._last_daily_date == date_str:
            return
        data = {
            "date": date_str,
            "bot_name": bot_name or self.bot_name,
            "trades": self._trade_count_since_report,
            "wins": self._wins_since_report,
            "losses": self._losses_since_report,
            "win_rate": self._wins_since_report / max(self._trade_count_since_report, 1),
            "pnl": f"${self._daily_pnl:.2f}" if abs(self._daily_pnl) > 0.01 else "$0.00",
            "best_trade": self._best_trade,
            "worst_trade": self._worst_trade,
            "regime": regime,
            "risk_level": risk_level,
            "exposure": exposure,
            "leverage": leverage,
            "active_trades": active_trades,
            "max_trades": max_trades,
        }
        msg = MessageFormatter.daily_summary(data)
        self._queue.append(Event("monitor", "daily_summary", "info", "Daily report", msg))
        self._reset_daily()
        self._last_daily_date = date_str

    def send_hourly_health(self, data: Dict):
        now = datetime.now(timezone.utc)
        if self._last_hourly_sent and (now - self._last_hourly_sent).total_seconds() < 3600:
            return
        data["bot_name"] = data.get("bot_name", self.bot_name)
        msg = MessageFormatter.bot_health(data)
        self._queue.append(Event("monitor", "bot_health", "info", "Bot health", msg))
        self._last_hourly_sent = now

    def _reset_daily(self):
        self._trade_count_since_report = 0
        self._wins_since_report = 0
        self._losses_since_report = 0
        self._daily_pnl = 0.0
        self._best_trade = None
        self._worst_trade = None
        self._best_profit = -999
        self._worst_profit = 999

    def flush(self):
        now = datetime.now(timezone.utc)
        critical = [e for e in self._queue if e.severity in ("critical", "emergency")]
        warnings = [e for e in self._queue if e.severity == "warning"]
        info = [e for e in self._queue if e.severity == "info"]

        new_queue = deque(maxlen=200)
        for e in self._queue:
            if e not in critical and e not in warnings and e not in info:
                new_queue.append(e)
        self._queue = new_queue

        for e in critical:
            self._send_now(e)

        for e in warnings:
            if self._rate_limiter.can_send(f"warning_{e.event_type}", 300):
                self._send_now(e)

        if info:
            rate_key = "info_digest"
            self._rate_limiter.reset(rate_key)
            if info and self._rate_limiter.can_send(rate_key, 600):
                if len(info) == 1:
                    self._send_now(info[0])
                else:
                    digest_lines = [f"━━━ DIGEST ({len(info)} events) ━━━"]
                    for e in info[:5]:
                        digest_lines.append(f"  - {e.title}")
                    if len(info) > 5:
                        digest_lines.append(f"  ... and {len(info) - 5} more")
                    digest_lines.append(MessageFormatter.SEP)
                    digest = "\n".join(digest_lines)
                    digest_event = Event("monitor", "digest", "info", "Digest", digest)
                    self._send_now(digest_event)
