import type { RefObject } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DashEnvelope,
  DashHostToPlayer,
  DashInitPayload,
  DashPlayerToHost,
  DashStatePayload,
  makeDashHostMessage,
} from "./dash-iframe-protocol";

type UseDashIframeOptions = {
  iframeRef: RefObject<HTMLIFrameElement>;
  playerOrigin: string;
};

export function useDashIframe({ iframeRef, playerOrigin }: UseDashIframeOptions) {
  const [isReady, setIsReady] = useState(false);
  const [lastState, setLastState] = useState<DashStatePayload | null>(null);
  const [lastEvent, setLastEvent] = useState<DashPlayerToHost | null>(null);
  const pending = useRef(new Map<string, (message: DashPlayerToHost) => void>());

  const post = useCallback(
    (message: DashEnvelope) => {
      const frameWindow = iframeRef.current?.contentWindow;
      if (!frameWindow) return false;
      frameWindow.postMessage(message, playerOrigin);
      return true;
    },
    [iframeRef, playerOrigin],
  );

  const send = useCallback(
    <TPayload,>(type: DashHostToPlayer["type"], payload?: TPayload) => {
      const id = crypto.randomUUID();
      const message = makeDashHostMessage(type, payload, id);
      return new Promise<DashPlayerToHost>((resolve, reject) => {
        pending.current.set(id, resolve);
        const ok = post(message);
        if (!ok) {
          pending.current.delete(id);
          reject(new Error("Iframe window not available"));
          return;
        }
        setTimeout(() => {
          if (pending.current.has(id)) {
            pending.current.delete(id);
            reject(new Error(`Timeout waiting for response to ${type}`));
          }
        }, 5000);
      });
    },
    [post],
  );

  useEffect(() => {
    function onMessage(event: MessageEvent<DashPlayerToHost>) {
      if (event.origin !== playerOrigin) return;
      const data = event.data;
      if (!data || data.source !== "dash-player" || data.version !== "1.0") return;

      setLastEvent(data);

      if (data.type === "dash.ready") {
        setIsReady(true);
        return;
      }

      if (data.type === "dash.state" && data.payload) {
        setLastState(data.payload);
      }

      if (data.id && pending.current.has(data.id)) {
        const resolve = pending.current.get(data.id)!;
        pending.current.delete(data.id);
        resolve(data);
      }
    }

    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [playerOrigin]);

  const api = useMemo(
    () => ({
      isReady,
      lastState,
      lastEvent,
      init: (payload: DashInitPayload) => send("dash.init", payload),
      loadScenario: (payload: DashInitPayload) => send("dash.loadScenario", payload),
      loadRecording: (recording: Record<string, unknown>) => send("dash.loadRecording", { recording }),
      play: () => send("dash.play"),
      pause: () => send("dash.pause"),
      reset: () => send("dash.reset"),
      seekTime: (seconds: number) => send("dash.seek", { mode: "time", value: seconds }),
      seekFrame: (frame: number) => send("dash.seek", { mode: "frame", value: frame }),
      setMode: (mode: "manual" | "autonomous") => send("dash.setMode", { mode }),
      setKeys: (keys: { forward?: boolean; backward?: boolean; left?: boolean; right?: boolean; brake?: boolean }) =>
        send("dash.setKeys", keys),
      clearKeys: () => send("dash.clearKeys"),
      getState: () => send("dash.getState"),
    }),
    [isReady, lastState, lastEvent, send],
  );

  return api;
}
