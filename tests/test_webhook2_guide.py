import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from scripts import post_webhook2_guide as guide


WEBHOOK = "https://discord.com/api/webhooks/123/fake-test-token"


def reply(message):
    return io.BytesIO(json.dumps(message).encode("utf-8"))


class GuideDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.stdout))
        self.enterContext(contextlib.redirect_stderr(self.stderr))
        self.enterContext(patch.dict(os.environ, {"DISCORD_WEBHOOK_URL_2": WEBHOOK}, clear=True))
        self.http = self.enterContext(patch.object(guide, "urlopen"))
        self.message = {"id": "123456", "channel_id": "789012", "embeds": guide.build_payload()["embeds"]}

    def test_dry_run_prints_valid_embed_without_network_or_secret(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(guide.main(["--dry-run"]), 0)
        self.http.assert_not_called()
        payload = json.loads(self.stdout.getvalue())
        self.assertEqual(len(payload["embeds"]), 1)
        self.assertIn("検出がない日は通知しません", payload["embeds"][0]["description"])
        self.assertLessEqual(len(payload["embeds"][0]["description"]), 4096)
        self.assertEqual(payload["allowed_mentions"], {"parse": []})

    def test_success_posts_exactly_once_then_reads_and_verifies_message(self):
        self.http.side_effect = [reply(self.message), reply(self.message)]
        self.assertEqual(guide.main([]), 0)
        self.assertEqual(self.http.call_count, 2)
        post = self.http.call_args_list[0].args[0]
        read = self.http.call_args_list[1].args[0]
        self.assertEqual(post.get_method(), "POST")
        self.assertEqual(post.full_url, WEBHOOK + "?wait=true")
        self.assertEqual(json.loads(post.data), guide.build_payload())
        self.assertEqual(read.get_method(), "GET")
        self.assertEqual(read.full_url, WEBHOOK + "/messages/123456")
        self.assertIn("VERIFIED: one embed", self.stdout.getvalue())
        self.assertNotIn(WEBHOOK, self.stdout.getvalue() + self.stderr.getvalue())

    def test_missing_secondary_does_not_fall_back_to_primary(self):
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL_2": "", "DISCORD_WEBHOOK_URL": "primary"}):
            self.assertEqual(guide.main([]), 2)
        self.http.assert_not_called()

    def test_http_failure_never_retries_or_logs_webhook_url(self):
        self.http.side_effect = HTTPError(WEBHOOK, 404, WEBHOOK, {}, None)
        self.assertEqual(guide.main([]), 1)
        self.assertEqual(self.http.call_count, 1)
        self.assertNotIn(WEBHOOK, self.stderr.getvalue())

    def test_ambiguous_timeout_never_retries(self):
        self.http.side_effect = URLError(WEBHOOK)
        self.assertEqual(guide.main([]), 1)
        self.assertEqual(self.http.call_count, 1)
        self.assertNotIn(WEBHOOK, self.stderr.getvalue())

    def test_missing_receipt_does_not_send_again(self):
        self.http.return_value = reply({})
        self.assertEqual(guide.main([]), 1)
        self.assertEqual(self.http.call_count, 1)

    def test_readback_failure_retains_message_id_and_does_not_repost(self):
        self.http.side_effect = [reply(self.message), URLError(WEBHOOK)]
        self.assertEqual(guide.main([]), 1)
        self.assertEqual(self.http.call_count, 2)
        self.assertIn("message_id=123456", self.stdout.getvalue())
        self.assertNotIn(WEBHOOK, self.stdout.getvalue() + self.stderr.getvalue())

    def test_changed_description_is_not_reported_as_verified(self):
        modified = dict(self.message, embeds=[dict(self.message["embeds"][0], description="wrong")])
        self.http.side_effect = [reply(self.message), reply(modified)]
        self.assertEqual(guide.main([]), 1)
        self.assertNotIn("VERIFIED", self.stdout.getvalue())
        self.assertEqual(self.http.call_count, 2)


if __name__ == "__main__":
    unittest.main()
