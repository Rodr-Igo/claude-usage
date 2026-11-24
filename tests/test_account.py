"""Tests for the account-vs-this-machine comparison (dashboard.py /api/account).

Every test uses a throwaway SQLite file and patches out the credential read and
the call to Anthropic, so nothing touches ~/.claude or the network.
"""

import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer
from pathlib import Path
from unittest import mock

import dashboard
import scanner
from scanner import get_db, init_db, insert_turns

TOKEN = "sk-ant-oat01-TEST-TOKEN-must-never-leak"
ACCT_A = {"id": "acct-company", "email": "team@example.com", "org": "Company"}
ACCT_B = {"id": "acct-personal", "email": "me@example.com", "org": None}
NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


def iso(dt):
    return dashboard._iso_utc(dt)


def turn(ts, model="claude-opus-4-8", msg=None, out=100):
    return {
        "session_id": "s1", "timestamp": iso(ts), "model": model,
        "input_tokens": 10, "output_tokens": out, "cache_read_tokens": 0,
        "cache_creation_tokens": 0, "tool_name": None, "cwd": "/tmp",
        "message_id": msg or "m-" + iso(ts) + model,
    }


def reading(five=None, week=None, resets5=None, resets7=None):
    out = {}
    if five is not None:
        out["five_hour"] = {"utilization": five, "resets_at": resets5}
    if week is not None:
        out["seven_day"] = {"utilization": week, "resets_at": resets7}
    return out


class TempDB(unittest.TestCase):
    def setUp(self):
        fd, name = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db_path = Path(name)
        self.conn = get_db(self.db_path)
        init_db(self.conn)
        dashboard._account_state.update(
            checked=None, reading=None, read_at=None, plan=None, account=None, error=None, detail=None)

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)
        dashboard._account_state.update(
            checked=None, reading=None, read_at=None, plan=None, account=None, error=None, detail=None)

    def rows(self, bucket="five_hour"):
        return self.conn.execute(
            "SELECT * FROM account_usage WHERE bucket = ? ORDER BY id", (bucket,)).fetchall()


class TestRecordReading(TempDB):
    def test_unchanged_reading_extends_last_seen(self):
        r = reading(five=10.0, resets5="2026-11-24T15:00:00.123456+00:00")
        dashboard.record_account_reading(self.conn, r, NOW)
        dashboard.record_account_reading(self.conn, r, NOW + timedelta(minutes=5))
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["first_seen"], iso(NOW))
        self.assertEqual(rows[0]["last_seen"], iso(NOW + timedelta(minutes=5)))

    def test_change_adds_row_in_same_window(self):
        dashboard.record_account_reading(self.conn, reading(five=10.0, resets5="2026-11-24T15:00:00Z"), NOW)
        # resets_at wobbles by a second: still the same window, same key
        dashboard.record_account_reading(
            self.conn, reading(five=14.0, resets5="2026-11-24T15:00:01Z"), NOW + timedelta(minutes=5))
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["resets_at"], rows[1]["resets_at"])

    def test_new_window_gets_new_key(self):
        dashboard.record_account_reading(self.conn, reading(five=40.0, resets5="2026-11-24T15:00:00Z"), NOW)
        dashboard.record_account_reading(
            self.conn, reading(five=40.0, resets5="2026-11-24T20:00:00Z"), NOW + timedelta(hours=3, minutes=1))
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["resets_at"], rows[1]["resets_at"])

    def test_skips_null_window_and_non_limit_keys(self):
        dashboard.record_account_reading(self.conn, {
            "five_hour": {"utilization": 0.0, "resets_at": None},
            "extra_usage": {"is_enabled": False},
            "iguana_necktie": None,
        }, NOW)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM account_usage").fetchone()[0], 0)


class TestAttribution(TempDB):
    RESETS = "2026-11-24T15:00:00Z"

    def record(self, pct, at):
        dashboard.record_account_reading(self.conn, reading(five=pct, resets5=self.RESETS), at)

    def key(self):
        return self.rows()[0]["resets_at"]

    def test_rises_split_by_local_activity(self):
        self.record(10.0, NOW)                           # already used: untracked
        self.record(15.0, NOW + timedelta(minutes=10))   # idle here -> other
        insert_turns(self.conn, [turn(NOW + timedelta(minutes=15))])
        self.conn.commit()
        self.record(25.0, NOW + timedelta(minutes=20))   # turn here -> active
        a = dashboard.attribute_window(self.conn, "five_hour", self.key(), None, 25.0)
        self.assertEqual(a["untracked"], 10.0)
        self.assertEqual(a["other"], 5.0)
        self.assertEqual(a["active"], 10.0)
        self.assertEqual(a["readings"], 3)
        self.assertEqual(a["tracked_since"], iso(NOW))

    def test_turn_just_before_previous_reading_counts_as_active(self):
        insert_turns(self.conn, [turn(NOW - timedelta(minutes=1))])
        self.conn.commit()
        self.record(10.0, NOW)
        self.record(12.0, NOW + timedelta(minutes=5))
        a = dashboard.attribute_window(self.conn, "five_hour", self.key(), None, 12.0)
        self.assertEqual(a["active"], 2.0)
        self.assertEqual(a["other"], 0.0)

    def test_turn_long_before_previous_reading_does_not_count(self):
        insert_turns(self.conn, [turn(NOW - timedelta(minutes=30))])
        self.conn.commit()
        self.record(10.0, NOW)
        self.record(12.0, NOW + timedelta(minutes=5))
        a = dashboard.attribute_window(self.conn, "five_hour", self.key(), None, 12.0)
        self.assertEqual(a["other"], 2.0)
        self.assertEqual(a["active"], 0.0)

    def test_non_claude_models_are_not_activity(self):
        self.record(10.0, NOW)
        insert_turns(self.conn, [turn(NOW + timedelta(minutes=2), model="gemma-3-27b")])
        self.conn.commit()
        self.record(12.0, NOW + timedelta(minutes=5))
        a = dashboard.attribute_window(self.conn, "five_hour", self.key(), None, 12.0)
        self.assertEqual(a["other"], 2.0)

    def test_family_window_only_counts_that_family(self):
        resets = "2026-11-30T15:00:00Z"
        dashboard.record_account_reading(
            self.conn, {"seven_day_opus": {"utilization": 10.0, "resets_at": resets}}, NOW)
        insert_turns(self.conn, [turn(NOW + timedelta(minutes=2), model="claude-sonnet-4-6")])
        self.conn.commit()
        dashboard.record_account_reading(
            self.conn, {"seven_day_opus": {"utilization": 12.0, "resets_at": resets}},
            NOW + timedelta(minutes=5))
        key = self.rows("seven_day_opus")[0]["resets_at"]
        a = dashboard.attribute_window(self.conn, "seven_day_opus", key, "opus", 12.0)
        self.assertEqual(a["other"], 2.0)   # a Sonnet turn isn't Opus activity

    def test_parts_scale_to_current_after_downward_revision(self):
        self.record(10.0, NOW)
        self.record(20.0, NOW + timedelta(minutes=5))   # +10 other
        self.record(16.0, NOW + timedelta(minutes=10))  # revised down
        a = dashboard.attribute_window(self.conn, "five_hour", self.key(), None, 16.0)
        self.assertAlmostEqual(a["other"] + a["active"] + a["untracked"], 16.0, places=1)

    def test_no_readings_returns_none(self):
        self.assertIsNone(dashboard.attribute_window(self.conn, "five_hour", self.RESETS, None, 5.0))


class TestGetAccountData(TempDB):
    def creds(self, **extra):
        c = {"accessToken": TOKEN, "subscriptionType": "max",
             "expiresAt": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000)}
        c.update(extra)
        return c

    def get(self, creds, fetch, force=False, account=None):
        account = account if account is not None else ACCT_A
        with mock.patch.object(dashboard, "read_oauth_credentials", return_value=creds), \
             mock.patch.object(dashboard, "read_oauth_account", return_value=account), \
             mock.patch.object(dashboard, "fetch_account_usage", side_effect=fetch) as f:
            data = dashboard.get_account_data(self.db_path, None, force=force, rescan=False)
        self.assertNotIn(TOKEN, json.dumps(data))
        return data, f

    def live_reading(self, five=30.0, week=55.0):
        now = datetime.now(timezone.utc)
        return reading(five=five, week=week,
                       resets5=iso(now + timedelta(hours=2)), resets7=iso(now + timedelta(days=3)))

    def test_disabled_by_env(self):
        with mock.patch.dict(os.environ, {"CLAUDE_USAGE_ACCOUNT": "off"}), \
             mock.patch.object(dashboard, "read_oauth_credentials") as rc, \
             mock.patch.object(dashboard, "read_oauth_account") as ra:
            data = dashboard.get_account_data(self.db_path, None, rescan=False)
        self.assertEqual(data["status"], "disabled")
        rc.assert_not_called()
        ra.assert_not_called()

    def test_no_login(self):
        data, f = self.get(None, None)
        self.assertEqual(data["status"], "no_login")
        f.assert_not_called()

    def test_expired_token_is_not_used(self):
        data, f = self.get(self.creds(expiresAt=1000), None)
        self.assertEqual(data["status"], "expired")
        f.assert_not_called()

    def test_auth_error(self):
        err = urllib.error.HTTPError(dashboard.ACCOUNT_USAGE_URL, 401, "Unauthorized", {}, io.BytesIO(b""))
        data, _ = self.get(self.creds(), err)
        self.assertEqual(data["status"], "auth")
        self.assertIn("HTTP 401", data["message"])

    def test_ok_includes_windows_local_usage_and_attribution(self):
        recent = datetime.now(timezone.utc) - timedelta(minutes=30)
        insert_turns(self.conn, [turn(recent, out=500)])
        self.conn.commit()
        data, f = self.get(self.creds(), lambda tok: self.live_reading())
        f.assert_called_once_with(TOKEN)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["plan"], "max")
        self.assertFalse(data["stale"])
        keys = [w["key"] for w in data["windows"]]
        self.assertEqual(keys, ["five_hour", "seven_day"])
        five = data["windows"][0]
        self.assertEqual(five["utilization"], 30.0)
        self.assertEqual(five["local"]["by_model"][0]["output"], 500)
        self.assertEqual(five["attribution"]["untracked"], 30.0)
        self.assertEqual(five["attribution"]["readings"], 1)

    def test_weekly_window_carries_surface_breakdown(self):
        def fetch(tok):
            r = self.live_reading()
            r["seven_day_breakdown"] = {"rows": [
                {"key": "claude_code", "display_name": "Claude Code", "percent": 80, "percent_of_limit": 44},
                {"key": "chat", "display_name": "Chats", "percent": 20, "percent_of_limit": 11},
                {"key": "junk"},
            ]}
            return r
        data, _ = self.get(self.creds(), fetch)
        five, week = data["windows"]
        self.assertIsNone(five["breakdown"])
        self.assertEqual([b["name"] for b in week["breakdown"]], ["Claude Code", "Chats"])
        self.assertEqual(week["breakdown"][1]["percent_of_limit"], 11)

    def test_cached_within_interval(self):
        _, f1 = self.get(self.creds(), lambda tok: self.live_reading())
        _, f2 = self.get(self.creds(), lambda tok: self.live_reading())
        self.assertEqual(f1.call_count, 1)
        self.assertEqual(f2.call_count, 0)

    def test_failure_after_success_serves_stale_reading(self):
        self.get(self.creds(), lambda tok: self.live_reading())
        dashboard._account_state["checked"] = None  # cache expired
        err = urllib.error.URLError("offline")
        data, _ = self.get(self.creds(), err)
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["stale"])
        self.assertIn("offline", data["warning"])
        self.assertEqual(data["windows"][0]["utilization"], 30.0)


class TestAccountSwitching(TempDB):
    RESETS = "2026-11-24T15:00:00Z"

    def test_readings_of_two_accounts_stay_apart(self):
        r = reading(five=10.0, resets5=self.RESETS)
        dashboard.record_account_reading(self.conn, r, NOW, "a")
        dashboard.record_account_reading(self.conn, reading(five=50.0, resets5=self.RESETS),
                                         NOW + timedelta(minutes=1), "b")
        dashboard.record_account_reading(self.conn, r, NOW + timedelta(minutes=2), "a")
        rows = self.conn.execute(
            "SELECT account_id, utilization, last_seen FROM account_usage ORDER BY id").fetchall()
        self.assertEqual([(x["account_id"], x["utilization"]) for x in rows], [("a", 10.0), ("b", 50.0)])
        self.assertEqual(rows[0]["last_seen"], iso(NOW + timedelta(minutes=2)))

    def test_login_log_is_run_length_encoded(self):
        dashboard.record_login(self.conn, ACCT_A, NOW)
        dashboard.record_login(self.conn, ACCT_A, NOW + timedelta(minutes=5))
        dashboard.record_login(self.conn, ACCT_B, NOW + timedelta(minutes=10))
        dashboard.record_login(self.conn, ACCT_A, NOW + timedelta(minutes=20))
        rows = self.conn.execute(
            "SELECT account_id, email, last_seen FROM account_logins ORDER BY id").fetchall()
        self.assertEqual([x["account_id"] for x in rows], ["acct-company", "acct-personal", "acct-company"])
        self.assertEqual(rows[0]["last_seen"], iso(NOW + timedelta(minutes=5)))
        self.assertEqual(rows[1]["email"], "me@example.com")

    def test_turns_made_on_another_account_are_not_activity(self):
        a = ACCT_A["id"]
        dashboard.record_login(self.conn, ACCT_A, NOW)
        dashboard.record_account_reading(self.conn, reading(five=10.0, resets5=self.RESETS), NOW, a)
        # switched to the personal account for a while and worked there
        dashboard.record_login(self.conn, ACCT_B, NOW + timedelta(minutes=10))
        insert_turns(self.conn, [turn(NOW + timedelta(minutes=20)), turn(NOW + timedelta(minutes=40))])
        dashboard.record_login(self.conn, ACCT_B, NOW + timedelta(minutes=50))
        self.conn.commit()
        # back on the company account: it rose 15% meanwhile -> the other dev
        dashboard.record_login(self.conn, ACCT_A, NOW + timedelta(minutes=60))
        dashboard.record_account_reading(self.conn, reading(five=25.0, resets5=self.RESETS),
                                         NOW + timedelta(minutes=60), a)
        key = self.rows()[0]["resets_at"]
        att = dashboard.attribute_window(self.conn, "five_hour", key, None, 25.0, a)
        self.assertEqual(att["other"], 15.0)
        self.assertEqual(att["active"], 0.0)
        # without the login log, the same turns would have counted as "active"
        self.conn.execute("DELETE FROM account_logins")
        att = dashboard.attribute_window(self.conn, "five_hour", key, None, 25.0, a)
        self.assertEqual(att["active"], 15.0)

    def test_local_usage_leaves_out_other_account_turns(self):
        dashboard.record_login(self.conn, ACCT_A, NOW)
        insert_turns(self.conn, [turn(NOW + timedelta(minutes=1))])
        dashboard.record_login(self.conn, ACCT_B, NOW + timedelta(minutes=10))
        insert_turns(self.conn, [turn(NOW + timedelta(minutes=15))])
        dashboard.record_login(self.conn, ACCT_B, NOW + timedelta(minutes=20))
        self.conn.commit()
        spans = dashboard.other_account_spans(self.conn, ACCT_A["id"], iso(NOW))
        local = dashboard.local_usage_since(self.conn, iso(NOW), None, spans)
        self.assertEqual(sum(m["turns"] for m in local["by_model"]), 1)
        self.assertEqual(local["excluded_turns"], 1)

    def test_accounts_summary_splits_turns_by_sign_in(self):
        dashboard.record_login(self.conn, ACCT_A, NOW)
        dashboard.record_account_reading(self.conn, reading(five=40.0, resets5=self.RESETS), NOW, ACCT_A["id"])
        insert_turns(self.conn, [turn(NOW + timedelta(minutes=1)), turn(NOW + timedelta(minutes=2))])
        dashboard.record_login(self.conn, ACCT_B, NOW + timedelta(minutes=10))
        dashboard.record_account_reading(self.conn, reading(five=5.0, resets5=self.RESETS),
                                         NOW + timedelta(minutes=10), ACCT_B["id"])
        insert_turns(self.conn, [turn(NOW + timedelta(minutes=12))])
        self.conn.commit()
        out = dashboard.accounts_summary(self.conn, ACCT_B["id"], NOW + timedelta(minutes=15))
        self.assertEqual([a["email"] for a in out], ["me@example.com", "team@example.com"])
        self.assertTrue(out[0]["current"])
        turns = {a["email"]: sum(m["turns"] for m in a["local"]["by_model"]) for a in out}
        self.assertEqual(turns, {"me@example.com": 1, "team@example.com": 2})
        self.assertEqual(out[1]["limits"]["five_hour"]["utilization"], 40.0)
        self.assertNotIn("id", out[0])


class TestRefreshAfterSwitch(TempDB):
    def creds(self):
        return {"accessToken": TOKEN, "subscriptionType": "pro",
                "expiresAt": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000)}

    def call(self, account, pct, force=False):
        now = datetime.now(timezone.utc)
        r = reading(five=pct, resets5=iso(now + timedelta(hours=2)))
        with mock.patch.object(dashboard, "read_oauth_credentials", return_value=self.creds()), \
             mock.patch.object(dashboard, "read_oauth_account", return_value=account), \
             mock.patch.object(dashboard, "fetch_account_usage", return_value=r) as f:
            data = dashboard.get_account_data(self.db_path, None, force=force, rescan=False)
        return data, f.call_count

    def test_switch_skips_the_cache_and_shows_the_new_account(self):
        data, calls = self.call(ACCT_A, 70.0)
        self.assertEqual((data["account"]["email"], calls), ("team@example.com", 1))
        data, calls = self.call(ACCT_A, 70.0)          # cached
        self.assertEqual(calls, 0)
        data, calls = self.call(ACCT_B, 5.0)           # /login to personal
        self.assertEqual(calls, 1)
        self.assertEqual(data["account"]["email"], "me@example.com")
        self.assertEqual(data["windows"][0]["utilization"], 5.0)
        self.assertEqual(len(data["accounts"]), 2)
        self.assertNotIn(TOKEN, json.dumps(data))

    def test_refresh_reads_again_after_a_few_seconds(self):
        self.call(ACCT_A, 70.0)
        dashboard._account_state["checked"] -= dashboard.ACCOUNT_FORCE_MIN_SECONDS + 1
        _, calls = self.call(ACCT_A, 72.0, force=True)
        self.assertEqual(calls, 1)


class TestReadOauthAccount(unittest.TestCase):
    def test_reads_oauth_account_from_config_dir(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, ".claude.json").write_text(json.dumps({"oauthAccount": {
                "accountUuid": "u-1", "emailAddress": "me@example.com", "organizationName": "Org"},
                "projects": {}}), encoding="utf-8")
            with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": d}):
                self.assertEqual(dashboard.read_oauth_account(),
                                 {"id": "u-1", "email": "me@example.com", "org": "Org"})

    def test_missing_or_unsigned_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": d}):
                self.assertIsNone(dashboard.read_oauth_account())
                Path(d, ".claude.json").write_text("{}", encoding="utf-8")
                self.assertIsNone(dashboard.read_oauth_account())


class TestScanLock(unittest.TestCase):
    def test_no_wait_returns_none_while_a_scan_runs(self):
        with scanner.SCAN_LOCK:
            self.assertIsNone(scanner.scan(projects_dirs=["/nonexistent"], db_path=Path("/nonexistent/x.db"),
                                           verbose=False, wait=False))


class TestAccountHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmpdir.name)
        (tmp / "projects").mkdir()
        cls._patches = [
            mock.patch.object(dashboard, "DB_PATH", tmp / "usage.db"),
            mock.patch.object(scanner, "DEFAULT_PROJECTS_DIRS", [tmp / "projects"]),
            mock.patch.object(dashboard, "read_oauth_credentials", return_value=None),
            mock.patch.object(dashboard, "read_oauth_account", return_value=None),
        ]
        for p in cls._patches:
            p.start()
        cls.server = HTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        for p in cls._patches:
            p.stop()
        cls._tmpdir.cleanup()

    def setUp(self):
        dashboard._account_state.update(
            checked=None, reading=None, read_at=None, plan=None, account=None, error=None, detail=None)

    def test_api_account_returns_json(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/account?refresh=1") as r:
            self.assertEqual(r.status, 200)
            self.assertIn("application/json", r.headers["Content-Type"])
            data = json.loads(r.read())
        self.assertEqual(data["status"], "no_login")


class TestAccountTemplate(unittest.TestCase):
    def test_card_and_loader_present(self):
        html = dashboard.HTML_TEMPLATE
        self.assertIn('id="sec-account"', html)
        self.assertIn('data-target="sec-account"', html)
        self.assertIn("function loadAccount(", html)
        self.assertIn("/api/account", html)


if __name__ == "__main__":
    unittest.main()
