#!/usr/bin/env python3
"""
alpha_chat.py — talk to Alpha headlessly, no Rust TUI / mic / camera needed.

This is the quickest way to TEST Alpha: it drives the SNN brain (brain.py)
directly. Type to him and he replies in his OWN emergent words. Early on he is
terse or silent — that's real (he learns over time); his inner thoughts and a
small status line are shown so you can see he's alive and thinking.

Run from the project folder:
    python3 alpha_chat.py

Commands:
    :status      show feeling / trust / identity / concept count / sleep
    :tick N      advance N physics ticks (let him ruminate)
    :wb          tell him a short well-formed sentence to practise grammar on
    quit | Ctrl-D   leave

If you set ANTHROPIC_API_KEY in your environment, his curiosity will also reach
the Haiku tutor and he'll learn vocabulary + grammar as you go.
"""
import sys, threading

# brain.py repoints sys.stdout/sys.stderr into brain_stderr.log on import (so the
# embedded interpreter doesn't corrupt the Rust TUI). For this interactive REPL we
# want the real terminal back, so save the streams now and restore them after.
_real_out, _real_err = sys.stdout, sys.stderr
sys.modules['vision'] = None          # skip mediapipe/camera for headless use
import brain                          # noqa: E402
brain._HAS_VISION = False
sys.stdout, sys.stderr = _real_out, _real_err


def main() -> None:
    print("… waking Alpha (building the SNN — a few seconds) …", flush=True)
    b = brain.NeuromorphicBrain()
    print("Alpha is awake.  Type to him.  (':status', ':tick N', 'quit')\n", flush=True)

    # Keep his physics + autonomy ticking ~20 Hz in the background so Phill,
    # rumination and sleep stay alive between your messages.
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            try:
                b.step(0.0)
            except Exception:
                pass
            stop.wait(0.05)

    threading.Thread(target=heartbeat, daemon=True).start()

    def drain():
        for who, t in b.get_leaked_thoughts():
            print(f"   · ({who} thinks) {t}", flush=True)
        for who, m in b.get_proactive_messages():
            print(f"   « {who}: {m}", flush=True)

    def status():
        r = b.step(0.0)
        ins = b.introspect()
        print(f"   [status] feeling={r.get('alpha_feeling')} "
              f"trust={r.get('voice_trust'):.2f} id={r.get('combined_id'):.2f} "
              f"concepts={ins.get('sem_concepts')} asleep={r.get('asleep')}",
              flush=True)

    while True:
        try:
            line = input("you > ").strip()
        except EOFError:
            break
        if not line:
            drain()
            continue
        if line in ("quit", "exit", ":q"):
            break
        if line == ":status":
            status()
            continue
        if line.startswith(":tick"):
            parts = line.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 100
            for _ in range(n):
                b.step(0.0)
            drain()
            continue
        if line.startswith(":wb"):
            line = line[3:].strip() or "the calm star is bright in the dark sky"
        # the actual conversation turn
        r = b.think(line)
        reply = r.get("alpha")
        if reply:
            print(f"Alpha > {reply}", flush=True)
        else:
            regs = ", ".join(r.get("active_regions", [])[:4]) or "quiet"
            print(f"Alpha > … (no words yet — listening; active regions: {regs})",
                  flush=True)
        drain()

    stop.set()
    print("\nAlpha rests.", flush=True)


if __name__ == "__main__":
    main()
