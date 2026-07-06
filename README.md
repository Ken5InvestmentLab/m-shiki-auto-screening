# M式自動スクリーニング

東証内国株式を対象に、Yahoo Financeの日足OHLCVから「上がる前の大口仕込み風反応」を日次検出し、判定日から直近7日以内に反応が出た銘柄をDiscord WebhookへEmbed通知します。

## 運用

- GASの時間主導トリガーから平日16:00 JSTにGitHub Actionsを起動します。
- GitHub Actions側のcron scheduleは使わず、手動実行またはGASからの`workflow_dispatch`で起動します。
- Discord Webhookはコードに保存せず、GitHub Secretsの`DISCORD_WEBHOOK_URL`に設定します。
- 通知は最大20件です。条件を満たす銘柄がない日も0件として要約通知します。
- EmbedタイトルからTradingViewへ直接移動できます。

## 検出ロジック

2つのレーンで検出します。

- `strong`: ローソク足品質込みでも強い反応。終値が日中レンジ上側に残り、長い上ヒゲではないもの。
- `quiet`: 価格がほぼ横ばいのまま、出来高/売買代金の単発反応が出たもの。

共通除外:

- 売買代金2,000万円未満
- 当日±6%超、5日±10%超、20日±18%超
- 52週高値圏
- 120日高値から5%未満の押し
- 直近上昇後/崩落後/長い上ヒゲ失速

## ローカル実行

```powershell
py -3 -m pip install -r requirements.txt
py -3 screen_big_money.py --dry-run --no-wait --allow-stale-data
```

Discordへ投稿する場合:

```powershell
$env:DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
py -3 screen_big_money.py --no-wait --allow-stale-data
```

## GitHub Actions設定

1. Repository Settings -> Secrets and variables -> Actions -> New repository secret で`DISCORD_WEBHOOK_URL`を設定します。
2. Actionsタブから`M式自動スクリーニング`を手動実行して、Discord投稿とTradingViewリンクを確認します。手動実行は既定で待機をスキップします。

## GAS起動設定

専用GASプロジェクト: https://script.google.com/d/12cN3Fd_mVrtC-0UIa9QlLP-jHtOu8gLgUZMsLxcVAFg5la9Pl99w8Tqt/edit

1. GitHub Fine-grained personal access tokenを作成し、このリポジトリに対してActions read/write権限を付与します。
2. GASプロジェクトのScript Propertiesに`GITHUB_TOKEN`としてトークンを保存します。
3. GASコードは[gas/Code.js](gas/Code.js)と[gas/appsscript.json](gas/appsscript.json)を`clasp push`で反映します。
4. GASエディタで`setupDailyTrigger`を1回実行し、16:00 JSTの日次トリガーを作成します。
5. GASの時間主導トリガーは秒単位の厳密実行ではありません。16:00前に起動した場合は、スクリプト内で16:00以降のリトライを予約します。
6. トリガー自体は毎日動きますが、スクリプト内で土日はスキップします。
