from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import screen_big_money as screening


PRIMARY = "https://primary.invalid/private-token-one"
SECONDARY = "https://secondary.invalid/private-token-two"


def make_result(count: int) -> screening.RunResult:
    candidates = [
        screening.Candidate(
            code=str(1000 + index), name=f"テスト銘柄{index}", market="プライム",
            lane="quiet", date="2026-11-02", score=95.0, close=100.0,
            day_ret=0.01, ret5=0.02, ret20=0.03, turnover=100_000_000,
            volume=1_000_000, turnover_ratio=6.0, volume_ratio=6.0,
            raw_pct=0.99, raw_to_max=0.8, qual_pct=0.99, qual_to_max=0.8,
            close_loc=0.7, upper_wick=0.1, pos20=0.2, pos252=0.3,
            dd120=-0.1, range_pct=0.02,
        )
        for index in range(count)
    ]
    return screening.RunResult(
        generated_at_jst="2026-11-02T16:00:00+09:00",
        screening_started_at_jst="2026-11-02T16:00:00+09:00",
        target_latest_date="2026-11-02", lookback_start_date="2026-11-02",
        jpx_list_date="20261031", universe_count=3800, yahoo_ok=3700,
        yahoo_errors=100, filtered_count=3700 - count,
        candidate_count=count, posted_count=count,
        latest_date_distribution={"2026-11-02": 3700},
        lane_counts={"quiet": count} if count else {}, error_counts={},
        candidates=candidates, notes=[],
    )


def response(status: int = 204) -> Mock:
    result = Mock(status_code=status)
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError("delivery rejected")
    return result


class DiscordNotificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.stdout))
        self.enterContext(contextlib.redirect_stderr(self.stderr))
        self.http = self.enterContext(patch.object(screening.requests, "post", return_value=response()))
        self.paths = (Path("unused.json"), Path("unused.csv"), Path("unused.html"))

    def send(self, count: int, secondary: str = SECONDARY, dry_run: bool = False) -> None:
        screening.post_to_discord(make_result(count), PRIMARY, self.paths, 0, dry_run, secondary)

    def payloads(self, url: str) -> list[dict]:
        return [
            json.loads(call.kwargs["data"]["payload_json"])
            for call in self.http.call_args_list if call.args[0] == url
        ]

    def test_detections_send_identical_summary_and_details_to_both(self) -> None:
        self.send(1)
        primary = self.payloads(PRIMARY)
        self.assertEqual(primary, self.payloads(SECONDARY))
        self.assertEqual(len(primary), 2)
        self.assertEqual(primary[0]["embeds"][0]["description"], "1銘柄を検出")
        self.assertEqual(primary[1]["embeds"][0]["title"], "#1 1000 テスト銘柄0")
        self.assertIn("TSE%3A1000", primary[1]["embeds"][0]["url"])

    def test_zero_detections_send_only_primary_summary(self) -> None:
        self.send(0)
        self.assertEqual(len(self.payloads(PRIMARY)), 1)
        self.assertEqual(self.payloads(PRIMARY)[0]["embeds"][0]["description"], "条件通過なし")
        self.assertEqual(self.payloads(SECONDARY), [])
        self.assertIn("skipped (no detections)", self.stdout.getvalue())

    def test_unconfigured_secondary_preserves_primary_for_zero_and_positive(self) -> None:
        for count in (0, 1):
            for secondary in ("", "  \t"):
                with self.subTest(count=count, secondary=secondary):
                    self.http.reset_mock()
                    self.send(count, secondary)
                    self.assertEqual(self.http.call_count, 1 if count == 0 else 2)
                    self.assertTrue(all(call.args[0] == PRIMARY for call in self.http.call_args_list))

    def test_default_argument_remains_compatible_with_single_webhook(self) -> None:
        screening.post_to_discord(make_result(1), PRIMARY, self.paths, 0, False)
        self.assertEqual(len(self.payloads(PRIMARY)), 2)
        self.assertEqual(self.payloads(SECONDARY), [])

    def test_twenty_candidates_are_ordered_in_batches_of_ten_per_destination(self) -> None:
        self.send(20)
        for url in (PRIMARY, SECONDARY):
            payloads = self.payloads(url)
            self.assertEqual([len(payload["embeds"]) for payload in payloads], [1, 10, 10])
            titles = [embed["title"] for payload in payloads[1:] for embed in payload["embeds"]]
            self.assertEqual(titles, [f"#{i + 1} {1000 + i} テスト銘柄{i}" for i in range(20)])

    def test_destinations_start_concurrently(self) -> None:
        rendezvous = threading.Barrier(2, timeout=5)

        def accept(url, *, data, **kwargs):
            if json.loads(data["payload_json"])["embeds"][0]["title"] == "M式自動スクリーニング":
                rendezvous.wait()
            return response()

        self.http.side_effect = accept
        self.send(1)
        self.assertEqual(len(self.payloads(PRIMARY)), 2)
        self.assertEqual(len(self.payloads(SECONDARY)), 2)

    def test_failure_of_either_destination_does_not_block_other(self) -> None:
        for failed_url, failed_name, healthy_url in (
            (PRIMARY, "DISCORD_WEBHOOK_URL", SECONDARY),
            (SECONDARY, "DISCORD_WEBHOOK_URL_2", PRIMARY),
        ):
            with self.subTest(failed_name=failed_name):
                self.http.reset_mock()

                def accept(url, **kwargs):
                    if url == failed_url:
                        raise requests.ConnectionError(f"secret request URL: {url}")
                    return response()

                self.http.side_effect = accept
                with self.assertRaises(RuntimeError) as raised:
                    self.send(1)
                self.assertEqual(str(raised.exception), f"Discord delivery failed: {failed_name}")
                self.assertEqual(len(self.payloads(healthy_url)), 2)
                self.assertEqual(len(self.payloads(failed_url)), 1)
                logs = self.stdout.getvalue() + self.stderr.getvalue() + str(raised.exception)
                self.assertNotIn(PRIMARY, logs)
                self.assertNotIn(SECONDARY, logs)

    def test_detail_failure_does_not_resend_summary_or_affect_other_destination(self) -> None:
        def accept(url, *, data, **kwargs):
            title = json.loads(data["payload_json"])["embeds"][0]["title"]
            if url == SECONDARY and title.startswith("#"):
                return response(500)
            return response()

        self.http.side_effect = accept
        with self.assertRaisesRegex(RuntimeError, "DISCORD_WEBHOOK_URL_2"):
            self.send(20)
        self.assertEqual(len(self.payloads(PRIMARY)), 3)
        self.assertEqual(len(self.payloads(SECONDARY)), 2)

    def test_both_failures_are_reported(self) -> None:
        self.http.return_value = response(404)
        with self.assertRaisesRegex(RuntimeError, "DISCORD_WEBHOOK_URL, DISCORD_WEBHOOK_URL_2"):
            self.send(1)
        self.assertEqual(self.http.call_count, 2)

    def test_rate_limit_retries_only_affected_destination(self) -> None:
        limited = response(429)
        limited.json.return_value = {"retry_after": 0}
        secondary_attempts = 0

        def accept(url, **kwargs):
            nonlocal secondary_attempts
            if url == SECONDARY:
                secondary_attempts += 1
                if secondary_attempts == 1:
                    return limited
            return response()

        self.http.side_effect = accept
        with patch.object(screening.time, "sleep") as sleep:
            self.send(1)
        sleep.assert_called_once_with(0.25)
        self.assertEqual(len(self.payloads(PRIMARY)), 2)
        self.assertEqual(len(self.payloads(SECONDARY)), 3)

    def test_dry_run_never_posts_with_or_without_detections(self) -> None:
        for count in (0, 1):
            with self.subTest(count=count):
                self.send(count, dry_run=True)
        self.http.assert_not_called()
        self.assertIn("Dry run destinations: DISCORD_WEBHOOK_URL, DISCORD_WEBHOOK_URL_2", self.stdout.getvalue())

    def test_identical_urls_do_not_duplicate_notifications(self) -> None:
        self.send(1, f"  {PRIMARY}  ")
        self.assertEqual(len(self.payloads(PRIMARY)), 2)

    def run_main(self, env: dict[str, str], *, dry_run: bool = False) -> int:
        with tempfile.TemporaryDirectory() as directory:
            argv = ["screen_big_money.py", "--no-wait", "--allow-stale-data", "--output-dir", directory]
            if dry_run:
                argv.append("--dry-run")
            with patch.dict(os.environ, env, clear=True), patch.object(screening.sys, "argv", argv), patch.object(
                screening, "retry_until_fresh", return_value=make_result(1)
            ):
                return screening.main()

    def test_main_reads_and_trims_both_environment_variables(self) -> None:
        status = self.run_main({"DISCORD_WEBHOOK_URL": f" {PRIMARY} ", "DISCORD_WEBHOOK_URL_2": f" {SECONDARY} "})
        self.assertEqual(status, 0)
        self.assertEqual(len(self.payloads(PRIMARY)), 2)
        self.assertEqual(len(self.payloads(SECONDARY)), 2)

    def test_main_works_without_secondary_secret(self) -> None:
        self.assertEqual(self.run_main({"DISCORD_WEBHOOK_URL": PRIMARY}), 0)
        self.assertEqual(len(self.payloads(PRIMARY)), 2)
        self.assertEqual(self.payloads(SECONDARY), [])

    def test_main_still_requires_primary_for_live_delivery(self) -> None:
        self.assertEqual(self.run_main({"DISCORD_WEBHOOK_URL_2": SECONDARY}), 2)
        self.http.assert_not_called()

    def test_main_dry_run_requires_no_secrets(self) -> None:
        self.assertEqual(self.run_main({}, dry_run=True), 0)
        self.http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
