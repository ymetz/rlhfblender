import * as THREE from 'three';
import PathPlanner from '../js/autonomy/path-planning/PathPlanner.js';
import LanePath from '../js/autonomy/LanePath.js';
import StaticObstacle from '../js/autonomy/StaticObstacle.js';
import DynamicObstacle from '../js/autonomy/DynamicObstacle.js';

function applyUtils() {
  Math.clamp = (number, min, max) => Math.max(min, Math.min(number, max));

  Math.wrapAngle = angle => {
    angle = angle % (Math.PI * 2);
    if (angle <= -Math.PI) return angle + Math.PI * 2;
    if (angle > Math.PI) return angle - Math.PI * 2;
    return angle;
  };

  THREE.Vector2.fromAngle = angle => new THREE.Vector2(Math.cos(angle), Math.sin(angle));

  THREE.Curve.prototype.getCurvatureAt = function(u) {
    let t2 = this.getUtoTmapping(u);

    const delta = 0.0001;
    let t1 = t2 - delta;
    let t3 = t2 + delta;

    if (t1 < 0) {
      t1 = 0;
      t2 = delta;
      t3 = 2 * delta;
    }

    if (t3 > 1) {
      t3 = 1;
      t2 = 1 - delta;
      t1 = 1 - 2 * delta;
    }

    const p1 = this.getPoint(t1);
    const p2 = this.getPoint(t2);
    const p3 = this.getPoint(t3);

    return (Math.atan2(p3.y - p2.y, p3.x - p2.x) - Math.atan2(p2.y - p1.y, p2.x - p1.x)) / p2.distanceTo(p1);
  };
}

function init() {
  globalThis.THREE = THREE;
  applyUtils();

  let pathPlanner;
  try {
    pathPlanner = new PathPlanner();
  } catch (e) {
    console.log('Error initializing path planner:');
    console.log(e);

    self.postMessage({ error: true });

    return;
  }

  self.onmessage = function(event) {
    const { config, vehiclePose, vehicleStation, lanePath, startTime, staticObstacles, dynamicObstacles, reset } = event.data;

    LanePath.hydrate(lanePath);
    staticObstacles.forEach(o => StaticObstacle.hydrate(o));
    dynamicObstacles.forEach(o => DynamicObstacle.hydrate(o));

    if (reset) pathPlanner.reset();

    pathPlanner.config = config;

    try {
      const { path, fromVehicleSegment, fromVehicleParams, latticeStartStation, dynamicObstacleGrid } = pathPlanner.plan(vehiclePose, vehicleStation, lanePath, startTime, staticObstacles, dynamicObstacles);

      self.postMessage({ path, fromVehicleSegment, fromVehicleParams, vehiclePose, vehicleStation, latticeStartStation, config, dynamicObstacleGrid });
    } catch (error) {
      console.log('PathPlannerWorker error');
      console.log(error);
    }
  };
}

init();
