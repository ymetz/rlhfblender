import Simulator from "./Simulator.js";
import StaticObstacle from "./autonomy/StaticObstacle.js";

function initSimulator() {
  window.simulator = new Simulator(document.getElementById('container'));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initSimulator, { once: true });
} else {
  initSimulator();
}
