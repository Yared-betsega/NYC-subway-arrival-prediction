# Running the recorder on a free Google Cloud VM

The VM records 24/7 and is the main copy of the data. Your PC is the backup.
Setup takes about 20 minutes and stays inside Google Cloud's free tier.

## 1. Account and project
1. Go to https://console.cloud.google.com and sign in with a personal Gmail account.
2. Accept the terms and set up billing. A card is required, but the e2-micro free tier costs $0.
3. Top bar → project picker → **New project** → name it `countdown` → Create.
4. **Billing → Budgets & alerts → Create budget:** set the amount to $1, with alerts at 50% and 100%. If anything ever costs money, you get an email.

## 2. Create the VM
Go to **Compute Engine → VM instances → Create instance**. Enable the Compute Engine API if it asks.

| Setting | Value |
|---|---|
| Name | `countdown-logger` |
| Region | `us-east1` (South Carolina); only us-east1, us-west1 and us-central1 are free |
| Machine type | E2 → **e2-micro** (2 vCPU, 1 GB memory) |
| Boot disk | Change → **Debian 12**, **Standard persistent disk**, **30 GB** |
| Firewall | Leave unchecked for now (the web app in Step 8 opens a port later) |

Click **Create**. The console may show a monthly estimate; the free-tier discount is applied on the bill.

## 3. Install the recorder
In the VM list, click **SSH** next to `countdown-logger`. A terminal opens in your browser.

```bash
sudo apt update && sudo apt install -y python3-venv
mkdir -p ~/countdown && cd ~/countdown
python3 -m venv .venv
.venv/bin/pip install requests gtfs-realtime-bindings pandas
```

Upload the two files using the SSH window's **⚙ → Upload file** button:
- `countdown/logger.py`
- `countdown/deploy/countdown-logger.service`

Both land in your home folder. Then run:

```bash
mv ~/logger.py ~/countdown/
sed -i "s/YOUR_USER/$(whoami)/g" ~/countdown-logger.service
sudo mv ~/countdown-logger.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now countdown-logger
```

## 4. Check it is running
```bash
systemctl status countdown-logger          # should say "active (running)"
tail -f ~/countdown/data/raw/logger.log    # a "poll ...: 8/8 feeds" line every 30 s; Ctrl+C to stop watching
```

Then test that it restarts on its own: `sudo reboot`, wait a minute, open SSH again and run `systemctl status countdown-logger`.

## 5. Weekly: copy the data off the VM
Each day's data is in its own folder, `~/countdown/data/raw/YYYY-MM-DD/`. Download finished days only, never today's folder while it is still being written.

```bash
cd ~/countdown/data/raw && tar czf ~/week.tgz 2026-10-31 2026-11-01   # list the finished days
```

Then use **⚙ → Download file** with the path `/home/<you>/week.tgz`, and put it on the shared Google Drive.

Disk check: run `df -h /`. Each day takes about 175 MB, so 30 GB is plenty for the whole project.
