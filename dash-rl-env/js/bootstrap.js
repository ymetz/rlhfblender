import "bulma/css/bulma.min.css";
import "@fortawesome/fontawesome-free/css/all.min.css";
import * as THREE from 'three';

async function start() {
  // Expose a mutable THREE object for legacy plugins/loaders that patch THREE.
  const legacyThree = { ...THREE };
  if (!legacyThree.Math && THREE.MathUtils) legacyThree.Math = THREE.MathUtils;

  window.THREE = legacyThree;
  globalThis.THREE = legacyThree;

  await import('./Utils.js');
  await import('./Dash.js');
}

start().catch(error => {
  console.error('Failed to bootstrap app', error);
});
