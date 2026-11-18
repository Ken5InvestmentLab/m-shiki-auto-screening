"""Manually publish the screening guide to webhook 2, then read it back."""
from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


DESCRIPTION = """東証の内国株式を対象に、日足の株価・出来高・売買代金から、**株価が大きく動く前の「仕込みを思わせる反応」**を探す自動スクリーニングです。

値動きが比較的落ち着いている銘柄の中から、通常より目立つ出来高・売買代金の変化を、次の3タイプで検出します。

🟢 **strong｜強反応**
出来高などの反応が強く、終値の位置や上ヒゲの短さにも条件を満たした銘柄。

🟡 **quiet｜静かな反応**
株価がほぼ横ばいの中で、出来高・売買代金に目立つ反応が出た銘柄。

🔵 **watch｜前兆**
quietより弱い反応も含め、横ばい圏で変化の兆しを拾った銘柄。

売買代金が少ない銘柄、高値圏の銘柄、直近の値動きが大きい銘柄、長い上ヒゲで失速した銘柄などは除外します。

**通知のタイミングと内容**

土日・祝日・年末年始を除く平日16時以降に処理を開始し、通常は当日に反応が出た銘柄を最大20件通知します。

通知には、銘柄名・反応タイプ・スコア・終値・騰落率・出来高・売買代金・通常比を掲載します。銘柄タイトルを押すとTradingViewのチャートを開けます。

**このチャンネルには、条件を満たす銘柄が見つかった日だけ通知します。検出がない日は通知しません。**

判定は日足データに基づく機械的なスクリーニングです。通知された銘柄のチャート確認や、監視候補の絞り込みにご活用ください。"""


def build_payload() -> dict:
    embed = {
        "title": "📊 大口仕込み自動スクリーニングについて",
        "description": DESCRIPTION,
        "color": 0x2DD4BF,
    }
    if len(embed["title"]) > 256 or len(DESCRIPTION) > 4096:
        raise ValueError("Guide exceeds Discord embed limits.")
    return {"embeds": [embed], "allowed_mentions": {"parse": []}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--edit-message-id", default=os.environ.get("GUIDE_MESSAGE_ID", ""))
    args = parser.parse_args(argv)
    if args.edit_message_id and not args.edit_message_id.isdigit():
        parser.error("--edit-message-id must contain only digits")
    payload = build_payload()
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL_2", "").strip()
    if not webhook_url:
        print("DISCORD_WEBHOOK_URL_2 is not configured.", file=sys.stderr)
        return 2
    parts = urlsplit(webhook_url)
    query = dict(parse_qsl(parts.query))
    query["wait"] = "true"
    post_url = urlunsplit(parts._replace(query=urlencode(query)))
    headers = {"Content-Type": "application/json", "User-Agent": "Big-Money-Screening-Guide/1.0"}
    method = "POST"
    if args.edit_message_id:
        post_url = urlunsplit(parts._replace(path=parts.path.rstrip("/") + "/messages/" + args.edit_message_id))
        try:
            with urlopen(Request(post_url, headers=headers), timeout=30) as response:
                current = json.load(response)
            expected = payload["embeds"][0]
            current_embeds = current.get("embeds", [])
            if (
                str(current.get("id")) != args.edit_message_id
                or len(current_embeds) != 1
                or current_embeds[0].get("title") not in (
                    "📊 M式自動スクリーニングについて", expected["title"]
                )
                or current_embeds[0].get("description") != expected["description"]
                or current_embeds[0].get("color") != expected["color"]
            ):
                raise ValueError("Existing message is not the expected guide.")
            # Preserve the existing embed and change only the requested title.
            payload["embeds"] = [dict(current_embeds[0], title=expected["title"])]
        except (HTTPError, URLError, OSError, ValueError, AttributeError, TypeError):
            print("Webhook2 existing guide could not be verified; no edit or new post.", file=sys.stderr)
            return 1
        method = "PATCH"
    request = Request(post_url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method=method)
    try:
        # Never retry a POST: a timeout may occur after Discord created the message.
        with urlopen(request, timeout=30) as response:
            message = json.load(response)
    except (HTTPError, URLError, OSError, ValueError):
        print(f"Webhook2 {method} failed or its receipt is unknown; no automatic resend.", file=sys.stderr)
        return 1

    if not isinstance(message, dict):
        print("Webhook2 response has no valid message receipt; do not resend automatically.", file=sys.stderr)
        return 1
    message_id = str(message.get("id", ""))
    channel_id = str(message.get("channel_id", ""))
    if not message_id.isdigit() or not channel_id.isdigit() or (args.edit_message_id and message_id != args.edit_message_id):
        print("Webhook2 response has no valid message receipt; do not resend automatically.", file=sys.stderr)
        return 1
    action = "updated" if args.edit_message_id else "created"
    print(f"Webhook2 {action} message_id={message_id} channel_id={channel_id}", flush=True)

    read_url = urlunsplit(parts._replace(path=parts.path.rstrip("/") + "/messages/" + message_id))
    try:
        with urlopen(Request(read_url, headers=headers), timeout=30) as response:
            saved = json.load(response)
        if not isinstance(saved, dict):
            raise ValueError("Invalid message response.")
        embeds = saved.get("embeds", [])
        expected = payload["embeds"][0]
        if (
            str(saved.get("id")) != message_id
            or str(saved.get("channel_id")) != channel_id
            or len(embeds) != 1
            or any(embeds[0].get(key) != value for key, value in expected.items())
        ):
            raise ValueError("Message differs from approved guide.")
    except (HTTPError, URLError, OSError, ValueError):
        print("Webhook2 message created, but read-back verification failed; do not repost.", file=sys.stderr)
        return 1

    print(f"VERIFIED: one embed; approved title, description and color match. message_id={message_id} channel_id={channel_id}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(f"Webhook2: guide sent and read back successfully. Message ID: `{message_id}`; channel ID: `{channel_id}`.\n")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
