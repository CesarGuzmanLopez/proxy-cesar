"""Compaction services for Sprint 4.

Pre-compaction: summarizes long inputs with a cheap model.
Continuous compaction: snapshots old turns when context exceeds trigger_pct.
External compaction detection: detects when the client (OpenCode) compacted.
"""
