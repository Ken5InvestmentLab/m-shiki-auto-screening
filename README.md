# M式自動スクリーニング

東証内国株式を対象に、Yahoo Financeの日足OHLCVから「上がる前の大口仕込み風反応」を日次検出し、判定日当日に反応が出た銘柄をDiscord WebhookへEmbed通知します。

## 運用

- GASの時間主導トリガーから平日16:00 JSTにGitHub Actionsを起動します。
- GitHub Actions側のcron scheduleは使わず、手動実行またはGASからの`workflow_dispatch`で起動します。
- Discord Webhookはコードに保存せず、GitHub Secretsの`DISCORD_WEBHOOK_URL`に設定します。
- 通知は最大20件です。条件を満たす銘柄がない日も0件として要約通知します。
- 任意の2つめのWebhookは`DISCORD_WEBHOOK_URL_2`に設定します。銘柄が検出された日だけ、1つめと同じ要約・銘柄詳細を並行して送信します。0件の日は2つめへ送信しません。
- 2つめが未設定・空欄なら従来どおり1つめだけに通知します。同じURLを両方に設定した場合は二重送信しません。片方の送信が失敗してももう片方への送信を試み、失敗があれば実行をエラーとして報告します。
- EmbedタイトルからTradingViewへ直接移動できます。

## 検出ロジック

3つのレーンで検出します。

- `strong`: ローソク足品質込みでも強い反応。終値が日中レンジ上側に残り、長い上ヒゲではないもの。
- `quiet`: 価格がほぼ横ばいのまま、出来高/売買代金の単発反応が出たもの。
- `watch`: `quiet`より少し弱いが、横ばい圏で前兆として拾いたい出来高/売買代金反応が出たもの。

共通除外:

- 売買代金2,000万円未満
- 当日騰落率が-2%未満または+4%超、5日±10%超、20日±18%超
- 事前20日高安値幅7%超、当日高安値幅4%超
- 52週高値圏
- 120日高値から5%未満の押し
- 直近上昇後/崩落後/長い上ヒゲ失速

## ローカル実行

```powershell
py -3 -m pip install -r requirements.txt
py -3 screen_entry.py --dry-run --no-wait --allow-stale-data
```

Discordへ投稿する場合:

```powershell
$env:DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
# 任意: 銘柄検出時だけ通知する2つめのWebhook
$env:DISCORD_WEBHOOK_URL_2 = "https://discord.com/api/webhooks/..."
py -3 screen_entry.py --no-wait --allow-stale-data
```

## GitHub Actions設定

1. Repository Settings -> Secrets and variables -> Actions -> New repository secret で`DISCORD_WEBHOOK_URL`を設定します。
2. Actionsタブから`M式自動スクリーニング`を手動実行して、Discord投稿とTradingViewリンクを確認します。手動実行は既定で待機をスキップします。
3. 過去期間をまとめて確認する手動実行だけ`lookback_days`を指定できます。通常の自動実行は既定値`0`のまま当日限定です。
4. 2つめを有効にするときは、同じ画面で`DISCORD_WEBHOOK_URL_2`という名前のRepository secretを追加し、新しいWebhook URLを保存します。コード変更やGAS設定の変更は不要で、次回実行から銘柄検出時だけ通知します。設定するまではSecretを作らなくて構いません。

## 通知処理のテスト

```powershell
py -3 -m unittest discover -s tests -v
```

外部へ送信せず、検出あり・0件・2つめ未設定・並行送信・片方の送信失敗・dry-runを確認します。`--dry-run`では両方のWebhookへ送信せず、送信予定の設定名と通知内容を表示します。

## Webhook2への仕様説明の手動送信

Actionsの`Webhook2 仕様説明を送信`を手動実行すると、`scripts/post_webhook2_guide.py`の説明文をWebhook2だけへ1件のEmbedとして送信し、作成したメッセージを読み戻して照合します。通常のスクリーニングとは独立し、自動では起動しません。

既存の仕様説明を修正する場合は、手動実行の`message_id`に対象メッセージIDを指定します。既存の説明文を確認してからタイトルをPATCHで編集し、同じメッセージを再読込して照合します。対象が見つからない場合は新規投稿せず停止します。ローカルでは`--edit-message-id`でも指定できます。

文面だけを確認する場合は`py -3 scripts/post_webhook2_guide.py --dry-run`を使います。送信は自動再試行しません。作成済みメッセージIDがログに残っている場合、確認処理の失敗を理由に再実行すると二重投稿になるため、既存メッセージを確認してください。

## GAS起動設定

専用GASプロジェクト: https://script.google.com/d/12cN3Fd_mVrtC-0UIa9QlLP-jHtOu8gLgUZMsLxcVAFg5la9Pl99w8Tqt/edit

1. GitHub Fine-grained personal access tokenを作成し、このリポジトリに対してActions read/write権限を付与します。
2. GASプロジェクトのScript Propertiesに`GITHUB_TOKEN`としてトークンを保存します。
3. GASコードは[gas/Code.js](gas/Code.js)と[gas/appsscript.json](gas/appsscript.json)を`clasp push`で反映します。
4. GASエディタで`setupDailyTrigger`を1回実行し、16:00 JSTの日次トリガーを作成します。
5. GASの時間主導トリガーは秒単位の厳密実行ではありません。16:00前に起動した場合は、スクリプト内で16:00以降のリトライを予約します。
6. トリガー自体は毎日確認を行いますが、GitHub Actionsの起動は土曜・日曜・国民の祝日・12月31日〜1月3日にスキップします。
7. 国民の祝日（振替休日・国民の休日を含む）は内閣府の公式CSVを参照し、取得失敗や対象年データの欠落時はワークフローを起動しません。
