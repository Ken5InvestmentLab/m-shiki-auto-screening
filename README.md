# M式自動スクリーニング

東証内国株式を対象に、Yahoo Financeの日足OHLCVから「上がる前の大口仕込み風反応」を日次検出し、Discord WebhookへEmbed通知します。

## 運用

- GitHub Actionsは平日15:50 JSTに起動し、ジョブ内で15:52:00 JSTまで待機してからスクリーニングします。
- Discord Webhookはコードに保存せず、GitHub Secretsの`DISCORD_WEBHOOK_URL`に設定します。
- 通知は最大20件です。条件を満たす銘柄が少ない日は0件から数件だけ通知します。
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
2. Actionsタブから`M式自動スクリーニング`を手動実行して、Discord投稿とTradingViewリンクを確認します。手動実行は既定で15:52待機をスキップします。
3. 平日は15:50 JSTにWorkflowが起動し、15:52 JSTまで待機してからスクリーニングします。
