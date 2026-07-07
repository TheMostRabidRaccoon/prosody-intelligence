# Running Prosody Intelligence on the swarm VM (always-on)

The engine historically ran only on the Conductor's Mac, so the swarm's
`prosody_analyze` tool failed with `no route to host` whenever the Mac was
asleep or the app wasn't open (an entire session on 2026-07-06 ran blind this
way). Fix: run a copy on the swarm VM itself under systemd. The Mac copy
stays for local/UI work; the VM copy is the one the swarm talks to.

Both transcription (OpenAI Whisper API) and TTS (ElevenLabs API) are hosted
calls — no GPU or local models, so the VM runs it fine headless.

## One-time setup (as `theconductor` on the VM)

```bash
sudo apt-get install -y ffmpeg            # pydub decode + compositor encode

cd ~ && git clone https://github.com/TheMostRabidRaccoon/prosody-intelligence
cd prosody-intelligence
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Keys: OPENAI_API_KEY + ELEVENLABS_API_KEY. src/app.py loads ~/.env first,
# then repo-local .env (repo wins). Easiest: put both keys in
# ~/prosody-intelligence/.env
cp /path/to/keys .env   # or create it by hand

sudo cp systemd/prosody.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now prosody.service
```

## Verify

```bash
curl -s localhost:5050/ >/dev/null && echo engine up
curl -s localhost:5000/prosody/status     # the swarm's own reachability check
```

The swarm needs **no config change**: `swarm_prosody.py` defaults to
`PROSODY_ENGINE_URL=http://localhost:5050`. If the swarm VM's `.env`
currently pins the Mac's LAN IP (e.g. `http://192.168.1.158:5050`), delete
that line and restart `swarm.service` so the default takes over.

## Notes

- `PROSODY_PORT` (default 5050) and `PROSODY_DEBUG` (default off) are env
  overrides; systemd runs with the Flask debug/reloader OFF.
- This serves Flask's dev server on the LAN, same as the Mac setup. Fine for
  a homelab; if the VM ever goes internet-facing, front it with gunicorn the
  same way the swarm's OPERATIONS.md prescribes for the main server.
- Updating: `cd ~/prosody-intelligence && git pull && sudo systemctl restart
  prosody.service` — same merged ≠ deployed ritual as the swarm.
