import { useEffect, useRef, useState } from "react";
import { useDashIframe } from "./useDashIframe";

type Props = {
  playerUrl: string;
  playerOrigin: string;
  scenarioName?: string;
  scenarioCode?: string;
};

export function DashIframeDemo({
  playerUrl,
  playerOrigin,
  scenarioName,
  scenarioCode,
}: Props) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const dash = useDashIframe({ iframeRef, playerOrigin });
  const [seekSeconds, setSeekSeconds] = useState(0);
  const [status, setStatus] = useState("waiting for dash.ready");
  const keyStateRef = useRef({
    forward: false,
    backward: false,
    left: false,
    right: false,
    brake: false,
  });

  useEffect(() => {
    if (!dash.isReady) return;

    void (async () => {
      try {
        await dash.init({
          scenarioName,
          scenarioCode,
          startMode: "manual",
          paused: true,
        });
        setStatus("initialized");
      } catch (err) {
        setStatus(`init failed: ${String(err)}`);
      }
    })();
  }, [dash, scenarioCode, scenarioName]);

  useEffect(() => {
    const timer = setInterval(() => {
      if (!dash.isReady) return;
      void dash.getState().catch(() => {});
    }, 500);
    return () => clearInterval(timer);
  }, [dash]);

  useEffect(() => {
    if (!dash.isReady) return;

    const updateKey = (key: string, isDown: boolean) => {
      switch (key) {
        case "w":
        case "W":
          keyStateRef.current.forward = isDown;
          return true;
        case "s":
        case "S":
          keyStateRef.current.backward = isDown;
          return true;
        case "a":
        case "A":
          keyStateRef.current.left = isDown;
          return true;
        case "d":
        case "D":
          keyStateRef.current.right = isDown;
          return true;
        case " ":
          keyStateRef.current.brake = isDown;
          return true;
        default:
          return false;
      }
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (!updateKey(event.key, true)) return;
      event.preventDefault();
      void dash.setKeys(keyStateRef.current);
    };

    const onKeyUp = (event: KeyboardEvent) => {
      if (!updateKey(event.key, false)) return;
      event.preventDefault();
      void dash.setKeys(keyStateRef.current);
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      void dash.clearKeys();
    };
  }, [dash]);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={() => void dash.play()} disabled={!dash.isReady}>
          Play
        </button>
        <button onClick={() => void dash.pause()} disabled={!dash.isReady}>
          Pause
        </button>
        <button onClick={() => void dash.reset()} disabled={!dash.isReady}>
          Reset
        </button>
        <input
          type="number"
          step={0.1}
          value={seekSeconds}
          onChange={(e) => setSeekSeconds(Number(e.target.value))}
          style={{ width: 100 }}
        />
        <button
          onClick={() => void dash.seekTime(seekSeconds)}
          disabled={!dash.isReady}
        >
          Seek Time
        </button>
      </div>

      <div>
        <strong>Status:</strong> {status}
      </div>
      <div>
        <strong>Latest state:</strong>{" "}
        <code>
          {dash.lastState ? JSON.stringify(dash.lastState) : "no state yet"}
        </code>
      </div>

      <iframe
        ref={iframeRef}
        src={playerUrl}
        title="Dash Player"
        style={{ width: "100%", height: 720, border: "1px solid #ccc" }}
      />
    </div>
  );
}
