# Dash Iframe Integration (React Reference)

This folder contains a host-side reference integration for embedding Dash in an iframe.

## Files

- `dash-iframe-protocol.ts`: message contract types and envelope helper.
- `useDashIframe.ts`: React hook for handshake + command/response messaging.
- `DashIframeDemo.tsx`: minimal demo component for init/play/pause/seek/state polling.

## Usage

```tsx
import { DashIframeDemo } from "./integration/DashIframeDemo";

export default function App() {
  return (
    <DashIframeDemo
      playerUrl="http://127.0.0.1:5173"
      playerOrigin="http://127.0.0.1:5173"
      scenarioName="rough_road"
      // or scenarioCode="PASTE_BASE64_SCENARIO_CODE_HERE"
    />
  );
}
```

## Player-side expectations

The Dash iframe page should:

1. Send `dash.ready` when loaded.
2. Accept host commands (`dash.init`, `dash.loadScenario`, `dash.play`, `dash.pause`, `dash.seek`, etc.).
3. Reply with `dash.ack` / `dash.nack` (echoing `id`).
4. Optionally emit `dash.state`, `dash.event`, `dash.frame`.

### Keyboard forwarding

Host can forward keyboard-like control states:

- `dash.setKeys` payload:
  `{ forward?: boolean, backward?: boolean, left?: boolean, right?: boolean, brake?: boolean }`
- `dash.clearKeys` to release all forwarded keys.

### Frame/time seek

Player supports `dash.seek` when a recording is loaded:

1. Load recording with `dash.loadRecording` (or include `recording` in `dash.init`).
2. Call:
   - `dash.seek` with `{ mode: "frame", value: <index> }`
   - or `{ mode: "time", value: <seconds> }`
3. Player replies with `dash.frame`.

## Security

- Host must verify `event.origin === playerOrigin`.
- Player should verify allowed host origin(s) before processing commands.
