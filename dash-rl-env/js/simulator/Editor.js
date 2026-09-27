import LanePath from "../autonomy/LanePath.js";
import StaticObstacle from "../autonomy/StaticObstacle.js";
import DynamicObstacleEditor from "./DynamicObstacleEditor.js";
import ScenarioManager from "./ScenarioManager.js";
import ShareManager from "./ShareManager.js";
import { formatDate } from "../Helpers.js";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";

const GROUND_PLANE = new THREE.Plane(new THREE.Vector3(0, 1, 0));

const NORMAL_OPACITY = 0.7;
const HOVER_OPACITY = 1;
const NORMAL_POINT_COLOR = 0x0088ff;
const HOVER_POINT_COLOR = 0x33ccff;
const NORMAL_STATIC_OBSTACLE_COLOR = 0xdd0000;
const HOVER_STATIC_OBSTACLE_COLOR = 0xdd3333;
const NORMAL_DYNAMIC_OBSTACLE_COLOR = 0xff8800;
const HOVER_DYNAMIC_OBSTACLE_COLOR = 0xffcc33;

const INITIAL_SPEED_FALLBACK = 20;
const SPEED_LIMIT_FALLBACK = 20;
const LANE_PREFERENCE_FALLBACK = +1;
const START_MODE_FALLBACK = 'autonomous';
const MAX_UNDO_ENTRIES = 200;

export default class Editor {
  constructor(canvas, camera, scene) {
    this.canvas = canvas;
    this.camera = camera;

    this.isEnabled = false;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.dragOffset = new THREE.Vector3();
    this.draggingPoint = null;
    this.pointIndex = 0;
    this.obstacleIndex = 0;
    this.previousSavedName = null;
    this.scenarioManager = new ScenarioManager(this);
    this.shareManager = new ShareManager();

    this.centerlineGeometry = new THREE.BufferGeometry();
    this.leftBoundaryGeometry = new THREE.BufferGeometry();
    this.rightBoundaryGeometry = new THREE.BufferGeometry();
    this.draggingObstaclePreview = null;

    this.group = new THREE.Group();
    this.group.renderOrder = 1;
    this.pointGroup = new THREE.Group();
    this.pointGroup.renderOrder = 2;
    this.obstacleGroup = new THREE.Group();
    this.obstacleGroup.renderOrder = 1;
    this.group.add(this.obstacleGroup);
    this.group.add(this.pointGroup);
    scene.add(this.group);

    this.lanePath = new LanePath();
    this.dynamicObstacleEditor = new DynamicObstacleEditor(() => this._commitHistory());

    this.editorPathButton = document.getElementById('editor-path');
    this.editorPathButton.addEventListener('click', e => this.changeEditMode('path'));
    this.editorObstaclesButton = document.getElementById('editor-obstacles');
    this.editorObstaclesButton.addEventListener('click', e => this.changeEditMode('staticObstacles'));
    this.editorDynamicObstaclesButton = document.getElementById('editor-dynamic-obstacles');
    this.editorDynamicObstaclesButton.addEventListener('click', e => this.changeEditMode('dynamicObstacles'));

    this.editorRoadBox = document.getElementById('editor-road-box');
    this.initialSpeedDom = document.getElementById('editor-initial-speed');
    this.speedLimitDom = document.getElementById('editor-speed-limit');
    this.laneLeftDom = document.getElementById('editor-lane-left');
    this.laneRightDom = document.getElementById('editor-lane-right');
    this.startModeManualDom = document.getElementById('editor-start-manual');
    this.startModeAutonomousDom = document.getElementById('editor-start-autonomous');

    this.laneLeftDom.addEventListener('click', () => {
      this._changeLanePreference(-1);
      this._commitHistory();
    });
    this.laneRightDom.addEventListener('click', () => {
      this._changeLanePreference(+1);
      this._commitHistory();
    });
    this.startModeManualDom.addEventListener('click', () => this.setStartMode('manual'));
    this.startModeAutonomousDom.addEventListener('click', () => this.setStartMode('autonomous'));
    this.initialSpeedDom.addEventListener('change', () => this._commitHistory());
    this.speedLimitDom.addEventListener('change', () => this._commitHistory());

    this.initialSpeedDom.value = INITIAL_SPEED_FALLBACK;
    this.speedLimitDom.value = SPEED_LIMIT_FALLBACK;
    this._changeLanePreference(LANE_PREFERENCE_FALLBACK);
    this.startMode = START_MODE_FALLBACK;
    this._changeStartMode(START_MODE_FALLBACK);

    this.statsRoadLength = document.getElementById('editor-stats-road-length');
    this.statsStaticObstacles = document.getElementById('editor-stats-static-obstacles');
    this.statsStation = document.getElementById('editor-stats-station');
    this.statsLatitude = document.getElementById('editor-stats-latitude');
    this.scenarioNameDom = document.getElementById('editor-scenario-name');
    this.scenarioSavedAtDom = document.getElementById('editor-scenario-saved-at');

    this.helpPath = document.getElementById('editor-help-path');
    this.helpStaticObstacles = document.getElementById('editor-help-static-obstacles');
    this.helpDynamicObstacles = document.getElementById('editor-help-dynamic-obstacles');

    this.changeEditMode('path');
    this.removeMode = false;

    canvas.addEventListener('mousedown', this.mouseDown.bind(this));
    canvas.addEventListener('mousemove', this.mouseMove.bind(this));
    canvas.addEventListener('mouseup', this.mouseUp.bind(this));
    canvas.addEventListener('contextmenu', e => this.isEnabled && e.preventDefault());

    const editorClearOptions = document.getElementById('editor-clear-options');
    document.getElementById('editor-clear').addEventListener('click', event => {
      event.stopPropagation();
      editorClearOptions.classList.toggle('is-hidden');
    });
    document.addEventListener('click', () => editorClearOptions.classList.add('is-hidden'));

    document.getElementById('editor-clear-obstacles').addEventListener('click', () => {
      this.clearStaticObstacles();
      this._commitHistory();
    });
    document.getElementById('editor-clear-dynamic-obstacles').addEventListener('click', () => this.dynamicObstacleEditor.clearDynamicObstacles());
    document.getElementById('editor-clear-path').addEventListener('click', () => {
      this.clearPath();
      this._commitHistory();
    });
    document.getElementById('editor-clear-all').addEventListener('click', () => {
      this.clearAll();
      this._commitHistory();
    });

    document.getElementById('editor-save').addEventListener('click', this.saveClicked.bind(this));
    document.getElementById('editor-load').addEventListener('click', this.loadClicked.bind(this));
    document.getElementById('editor-share').addEventListener('click', this.shareClicked.bind(this));
    this.undoButton = document.getElementById('editor-undo');
    this.undoButton.addEventListener('click', this.undo.bind(this));

    document.addEventListener('keydown', this.keyDown.bind(this));
    document.addEventListener('keyup', this.keyUp.bind(this));

    this.undoHistory = [];
    this.isRestoringHistory = false;
    this.isApplyingScenario = false;

    const resolution = new THREE.Vector2(this.canvas.clientWidth, this.canvas.clientHeight);
    this.centerlineObject = new Line2(
      new LineGeometry(),
      new LineMaterial({
        color: new THREE.Color(0x004488),
        linewidth: 8,
        resolution: resolution,
        worldUnits: false,
        depthTest: false,
        depthWrite: false
      })
    );
    this.centerlineObject.renderOrder = 1;
    this.group.add(this.centerlineObject);

    this.leftBoundaryObject = new Line2(
      new LineGeometry(),
      new LineMaterial({
        color: new THREE.Color(0xff40ff),
        linewidth: 4,
        resolution: resolution,
        worldUnits: false,
        depthTest: false,
        depthWrite: false,
        transparent: true,
        opacity: 0.7
      })
    );
    this.leftBoundaryObject.renderOrder = 1;
    this.group.add(this.leftBoundaryObject);

    this.rightBoundaryObject = new Line2(
      new LineGeometry(),
      new LineMaterial({
        color: new THREE.Color(0xff40ff),
        linewidth: 4,
        resolution: resolution,
        worldUnits: false,
        depthTest: false,
        depthWrite: false,
        transparent: true,
        opacity: 0.7
      })
    );
    this.rightBoundaryObject.renderOrder = 1;
    this.group.add(this.rightBoundaryObject);

    window.addEventListener('resize', () => {
      // Use setTimeout to queue the resolution update after the canvas is reflowed.
      // This gets around some weirdness noticed when opening and closing Chrome Developer Tools.
      setTimeout(() => {
        const resolution = new THREE.Vector2(this.canvas.clientWidth, this.canvas.clientHeight);
        this.centerlineObject.material.resolution.copy(resolution);
        this.leftBoundaryObject.material.resolution.copy(resolution);
        this.rightBoundaryObject.material.resolution.copy(resolution);
      }, 0);
    });

    this._commitHistory();
  }

  get enabled() {
    return this.isEnabled;
  }

  set enabled(e) {
    this.isEnabled = e;
    this.pointGroup.visible = this.obstacleGroup.visible = !!this.isEnabled
  }

  get staticObstacles() {
    return this.obstacleGroup.children.map(o => new StaticObstacle(new THREE.Vector2(o.position.x, o.position.z), -o.rotation.z, o.userData.width, o.userData.height));
  }

  get dynamicObstacles() {
    return this.dynamicObstacleEditor.collectDynamicObstacles();
  }

  get initialSpeed() {
    let speed = parseFloat(this.initialSpeedDom.value);
    if (Number.isNaN(speed) || speed < 0)
      speed = 0;

    return Number.isNaN(speed) || speed < 0 ? INITIAL_SPEED_FALLBACK : speed;
  }

  get speedLimit() {
    let limit = parseFloat(this.speedLimitDom.value);
    if (Number.isNaN(limit) || limit < 0)
      limit = 0;

    return Number.isNaN(limit) || limit < 0 ? SPEED_LIMIT_FALLBACK : limit;
  }

  scenarioToJSON() {
    const trunc = n => +n.toFixed(5);

    const json = {
      p: Array.prototype.concat.apply([], this.lanePath.anchors.map(a => [trunc(a.x), trunc(a.y)])),
      s: this.staticObstacles.map(o => o.toJSON()),
      d: this.dynamicObstacleEditor.toJSON(),
      l: Number(this.lanePath.arcLength.toFixed(3)),
      c: {
        s: this.initialSpeedDom.value,
        sl: this.speedLimitDom.value,
        lp: this.lanePreference,
        sm: this.startMode === 'manual' ? 'm' : 'a'
      },
      v: 1
    };

    return json;
  }

  loadJSON(json) {
    if (json.p === undefined || json.p.length % 2 != 0) {
      throw new Error('Incomplete lane path.');
    }

    this.isApplyingScenario = true;
    try {
      this.clearAll();

      this.lanePath = new LanePath();
      for (let i = 0; i < json.p.length; i += 2) {
        this.addPoint(new THREE.Vector2(json.p[i], json.p[i + 1]), false);
      }
      this.lanePath.resampleAll();
      this.rebuildPathGeometry();

      json.s.forEach(o => {
        const staticObstacle = StaticObstacle.fromJSON(o);
        this.addStaticObstacle(new THREE.Vector3(staticObstacle.pos.x, 0, staticObstacle.pos.y), staticObstacle.width, staticObstacle.height, staticObstacle.rot)
      });

      this.dynamicObstacleEditor.loadJSON(json.d);

      let initialSpeed = INITIAL_SPEED_FALLBACK;
      let speedLimit = SPEED_LIMIT_FALLBACK;
      try { initialSpeed = json.c.s; } catch (e) { }
      try { speedLimit = json.c.sl; } catch (e) { }

      this.initialSpeedDom.value = initialSpeed;
      this.speedLimitDom.value = speedLimit;

      let lanePreference = LANE_PREFERENCE_FALLBACK;
      try {
        if (typeof(json.c.lp) === 'number')
          lanePreference = Math.sign(json.c.lp) || LANE_PREFERENCE_FALLBACK;
      } catch (e) { }

      this._changeLanePreference(lanePreference);

      let startMode = START_MODE_FALLBACK;
      try {
        if (json.c.sm === 'm' || json.c.sm === 'manual')
          startMode = 'manual';
        else if (json.c.sm === 'a' || json.c.sm === 'autonomous')
          startMode = 'autonomous';
      } catch (e) { }

      this._changeStartMode(startMode);
    } finally {
      this.isApplyingScenario = false;
    }

    this._commitHistory();
  }

  update() {
    if (!this.isEnabled) return;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersection = this.raycaster.ray.intersectPlane(GROUND_PLANE, new THREE.Vector3());

    const [station, latitude, _around] = this.lanePath.stationLatitudeFromPosition(new THREE.Vector2(intersection.x, intersection.z));
    this.statsStation.textContent = (station || 0).toFixed(1);
    this.statsLatitude.textContent = (latitude || 0).toFixed(1);

    if (this.draggingPoint) {
      if (intersection != null) {
        this.updatePoint(this.draggingPoint, intersection.clone().add(this.dragOffset));
        this.rebuildPathGeometry();
      }
    } else if (this.draggingObstacle) {
      if (intersection !== null) {
        if (this.draggingObstacle === true) {
          if (this.draggingObstaclePreview) this.group.remove(this.draggingObstaclePreview);

          const [center, width, height] = this._dimensionsFromRect(this.dragOffset, intersection);

          this.draggingObstaclePreview = new THREE.Mesh(
            new THREE.PlaneGeometry(width, height),
            new THREE.MeshBasicMaterial({ color: NORMAL_STATIC_OBSTACLE_COLOR, depthTest: false, transparent: true, opacity: 0.4 })
          );
          this.draggingObstaclePreview.rotation.x = -Math.PI / 2;
          this.draggingObstaclePreview.position.copy(center);
          this.group.add(this.draggingObstaclePreview);
        } else {
          this.draggingObstacle.position.copy(intersection.clone().add(this.dragOffset));
        }
      }
    } else if (this.rotatingObstacle) {
      const rotation = (this.dragOffset.x - this.mouse.x) * 2 *  Math.PI;
      this.rotatingObstacle.rotation.z = Math.wrapAngle(rotation + this.initialObstacleRotation);
    } else {
      this.pointGroup.children.forEach(p => {
        p.material.color.set(NORMAL_POINT_COLOR)
        p.material.opacity = NORMAL_OPACITY;
      });

      this.obstacleGroup.children.forEach(o => {
        o.material.color.set(NORMAL_STATIC_OBSTACLE_COLOR)
        o.material.opacity = NORMAL_OPACITY;
      });

      this.canvas.classList.remove('editor-grab', 'editor-grabbing', 'editor-removing');

      if (this.editMode == 'path' && this.pointGroup.children.length > 0) {
        let picked = null;
        this.raycaster.intersectObjects(this.pointGroup.children).forEach(p => {
          if (picked === null || p.object.userData.index > picked.object.userData.index) picked = p;
        });

        if (picked) {
          picked.object.material.color.set(HOVER_POINT_COLOR);
          picked.object.material.opacity = HOVER_OPACITY;

          if (this.removeMode)
            this.canvas.classList.add('editor-removing');
          else
            this.canvas.classList.add('editor-grab');
        }
      } else if (this.editMode == 'staticObstacles' && this.obstacleGroup.children.length > 0) {
        let picked = null;
        this.raycaster.intersectObjects(this.obstacleGroup.children).forEach(o => {
          if (picked === null || o.object.userData.index > picked.object.userData.index) picked = o;
        });

        if (picked) {
          picked.object.material.color.set(HOVER_STATIC_OBSTACLE_COLOR);
          picked.object.material.opacity = HOVER_OPACITY;

          if (this.removeMode)
            this.canvas.classList.add('editor-removing');
          else
            this.canvas.classList.add('editor-grab');
        }
      }
    }
  }

  changeEditMode(mode) {
    this.editorPathButton.classList.add('is-outlined');
    this.editorObstaclesButton.classList.add('is-outlined');
    this.editorDynamicObstaclesButton.classList.add('is-outlined');
    this.editorPathButton.classList.remove('is-selected');
    this.editorObstaclesButton.classList.remove('is-selected');
    this.editorDynamicObstaclesButton.classList.remove('is-selected');
    this.editorRoadBox.classList.add('is-hidden');
    this.helpPath.classList.add('is-hidden');
    this.helpStaticObstacles.classList.add('is-hidden');
    this.helpDynamicObstacles.classList.add('is-hidden');

    if (mode == 'path') {
      this.editMode = 'path';
      this.editorPathButton.classList.remove('is-outlined');
      this.editorPathButton.classList.add('is-selected');
      this.editorRoadBox.classList.remove('is-hidden');
      this.helpPath.classList.remove('is-hidden');
      this.dynamicObstacleEditor.disable();
    } else if (mode == 'staticObstacles') {
      this.editMode = 'staticObstacles';
      this.editorObstaclesButton.classList.remove('is-outlined');
      this.editorObstaclesButton.classList.add('is-selected');
      this.helpStaticObstacles.classList.remove('is-hidden');
      this.dynamicObstacleEditor.disable();
    } else {
      this.editMode = 'dynamicObstacles';
      this.editorDynamicObstaclesButton.classList.remove('is-outlined');
      this.editorDynamicObstaclesButton.classList.add('is-selected');
      this.helpDynamicObstacles.classList.remove('is-hidden');
      this.dynamicObstacleEditor.enable();
    }
  }

  addStaticObstacle(center, width, height, rotation = 0) {
    const obstacle = new THREE.Mesh(
      new THREE.PlaneGeometry(width, height),
      new THREE.MeshBasicMaterial({ color: NORMAL_STATIC_OBSTACLE_COLOR, depthTest: false, transparent: true, opacity: NORMAL_OPACITY })
    );
    obstacle.rotation.x = -Math.PI / 2;
    obstacle.rotation.z = -Math.wrapAngle(rotation);
    obstacle.position.copy(center);
    obstacle.userData = { index: this.obstacleIndex++, width: width, height: height };

    this.obstacleGroup.add(obstacle);
    this.statsStaticObstacles.textContent = this.obstacleGroup.children.length;
  }

  removeStaticObstacle(obstacle) {
    this.obstacleGroup.remove(obstacle);
    this.statsStaticObstacles.textContent = this.obstacleGroup.children.length;
  }

  clearStaticObstacles() {
    this.group.remove(this.obstacleGroup);
    this.obstacleGroup = new THREE.Group();
    this.obstacleGroup.renderOrder = 1;
    this.group.add(this.obstacleGroup);
    this.obstacleIndex = 0;
    this.statsStaticObstacles.textContent = 0;
  }

  clearAll() {
    this.clearPath();
    this.clearStaticObstacles();
    this.dynamicObstacleEditor.clearDynamicObstacles();
  }

  rebuildPathGeometry() {
    const toLinePositions = points => {
      const positions = new Array(points.length * 3);
      for (let i = 0; i < points.length; i++) {
        positions[i * 3] = points[i].x;
        positions[i * 3 + 1] = 0;
        positions[i * 3 + 2] = points[i].y;
      }
      return positions;
    };

    if (this.lanePath.anchors.length > 1) {
      this.centerlineGeometry = new LineGeometry();
      this.centerlineGeometry.setPositions(toLinePositions(this.lanePath.centerline));
      this.centerlineObject.geometry.dispose();
      this.centerlineObject.geometry = this.centerlineGeometry;

      this.leftBoundaryGeometry = new LineGeometry();
      this.leftBoundaryGeometry.setPositions(toLinePositions(this.lanePath.leftBoundary));
      this.leftBoundaryObject.geometry.dispose();
      this.leftBoundaryObject.geometry = this.leftBoundaryGeometry;

      this.rightBoundaryGeometry = new LineGeometry();
      this.rightBoundaryGeometry.setPositions(toLinePositions(this.lanePath.rightBoundary));
      this.rightBoundaryObject.geometry.dispose();
      this.rightBoundaryObject.geometry = this.rightBoundaryGeometry;
    } else {
      this.centerlineObject.geometry.dispose();
      this.centerlineObject.geometry = new LineGeometry();

      this.leftBoundaryObject.geometry.dispose();
      this.leftBoundaryObject.geometry = new LineGeometry();

      this.rightBoundaryObject.geometry.dispose();
      this.rightBoundaryObject.geometry = new LineGeometry();
    }

    this.statsRoadLength.textContent = this.lanePath.arcLength.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  }

  addPoint(pos, resample = true) {
    const point = new THREE.Mesh(
      new THREE.CircleGeometry(1, 32),
      new THREE.MeshBasicMaterial({
        color: NORMAL_POINT_COLOR,
        depthTest: false,
        transparent: true,
        opacity: NORMAL_OPACITY
      })
    );
    point.rotation.x = -Math.PI / 2;
    point.position.set(pos.x, 0, pos.y);
    point.userData = { index: this.pointIndex++ };

    this.lanePath.addAnchor(pos, resample);
    this.pointGroup.add(point);

    return point;
  }

  updatePoint(object, pos) {
    object.position.copy(pos);
    this.lanePath.updateAnchor(object.userData.index, new THREE.Vector2(pos.x, pos.z));
  }

  removePoint(object) {
    const index = object.userData.index;

    this.pointGroup.remove(object);
    this.pointGroup.children.forEach(p => {
      if (p.userData.index > index) p.userData.index--;
    });
    this.pointIndex--;

    this.lanePath.removeAnchor(index);
  }

  clearPath() {
    this.group.remove(this.pointGroup);
    this.pointGroup = new THREE.Group();
    this.pointGroup.renderOrder = 2;
    this.group.add(this.pointGroup);
    this.pointIndex = 0;

    this.lanePath = new LanePath();
    this.rebuildPathGeometry();

    this.initialSpeedDom.value = INITIAL_SPEED_FALLBACK;
    this.speedLimitDom.value = SPEED_LIMIT_FALLBACK;
  }

  keyDown(event) {
    if (event.repeat) return;

    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() == 'z') {
      this.undo();
      event.preventDefault();
      return;
    }

    if (this.editMode != 'path' && this.editMode != 'staticObstacles') return;

    if (event.key == 'Shift') {
      this.removeMode = true;
      this.canvas.classList.add('editor-pointing');
      event.preventDefault();
    } else if (event.key == 'Control' && this.editMode == 'staticObstacles') {
      this.rotateMode = true;
      this.canvas.classList.add('editor-pointing');
      event.preventDefault();
    }
  }

  keyUp(event) {
    if (event.key == 'Shift') {
      this.removeMode = false;
      this.canvas.classList.remove('editor-pointing', 'editor-removing');
    } else if (event.key == 'Control') {
      this.rotateMode = false;
      this.canvas.classList.remove('editor-pointing', 'editor-grabbing');
    }
  }

  mouseDown(event) {
    if (!this.isEnabled || event.button != 0) return;

    this.mouse.x = (event.offsetX / this.canvas.clientWidth) * 2 - 1;
    this.mouse.y = -(event.offsetY / this.canvas.clientHeight) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);

    if (this.editMode == 'path') {
      let picked = null;
      this.raycaster.intersectObjects(this.pointGroup.children).forEach(p => {
        if (picked === null || p.object.userData.index > picked.object.userData.index) picked = p;
      });

      if (picked) {
        if (this.removeMode) {
          this.removePoint(picked.object);
          this.rebuildPathGeometry();
          this._commitHistory();
        } else {
          this.canvas.classList.remove('editor-grab');
          this.canvas.classList.add('editor-grabbing');

          this.draggingPoint = picked.object;
          this.dragOffset.copy(picked.object.position).sub(picked.point);
        }
      } else if (!this.removeMode) {
        const intersection = this.raycaster.ray.intersectPlane(GROUND_PLANE, new THREE.Vector3());
        if (intersection != null) {
          this.addPoint(new THREE.Vector2(intersection.x, intersection.z));
          this.rebuildPathGeometry();
          this._commitHistory();
        }
      }
    } else if (this.editMode == 'staticObstacles') {
      let picked = null;
      this.raycaster.intersectObjects(this.obstacleGroup.children).forEach(o => {
        if (picked === null || o.object.userData.index > picked.object.userData.index) picked = o;
      });

      if (picked) {
        if (this.removeMode) {
          this.removeStaticObstacle(picked.object);
          this._commitHistory();
        } else {
          this.canvas.classList.remove('editor-grab');
          this.canvas.classList.add('editor-grabbing');

          if (this.rotateMode) {
            this.rotatingObstacle = picked.object;
            this.initialObstacleRotation = picked.object.rotation.z;
            this.dragOffset.set(this.mouse.x, this.mouse.y, 0);
          } else {
            this.draggingObstacle = picked.object;
            this.dragOffset.copy(picked.object.position).sub(picked.point);
          }
        }
      } else if (!this.removeMode && !this.rotateMode) {
        const intersection = this.raycaster.ray.intersectPlane(GROUND_PLANE, new THREE.Vector3());
        if (intersection != null) {
          this.draggingObstacle = true;
          this.dragOffset.copy(intersection);
        }
      }
    }
  }

  mouseMove(event) {
    this.mouse.x = (event.offsetX / this.canvas.clientWidth) * 2 - 1;
    this.mouse.y = -(event.offsetY / this.canvas.clientHeight) * 2 + 1;
  }

  mouseUp(event) {
    if (!this.isEnabled || event.button != 0) return;

    const movedPoint = this.draggingPoint !== null;
    const movedObstacle = this.draggingObstacle !== null && this.draggingObstacle !== true;

    if (this.draggingObstacle === true) {
      this.group.remove(this.draggingObstaclePreview);
      this.draggingObstaclePreview = null;

      this.mouse.x = (event.offsetX / this.canvas.clientWidth) * 2 - 1;
      this.mouse.y = -(event.offsetY / this.canvas.clientHeight) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);

      const intersection = this.raycaster.ray.intersectPlane(GROUND_PLANE, new THREE.Vector3());
      if (intersection != null) {
        const [center, width, height] = this._dimensionsFromRect(this.dragOffset, intersection);
        this.addStaticObstacle(center, width, height);
        this._commitHistory();
      }
    }

    if (movedPoint || movedObstacle)
      this._commitHistory();

    this.draggingPoint = null;
    this.draggingObstacle = null;
    this.rotatingObstacle = null;
    this.canvas.classList.remove('editor-grab', 'editor-grabbing');
  }

  updateSavedInfo(name, savedAt) {
    this.previousSavedName = name || null;

    name = name || 'Untitled';
    savedAt = savedAt || 'Unsaved';

    this.scenarioNameDom.textContent = name;
    this.scenarioNameDom.title = name;
    this.scenarioSavedAtDom.textContent = savedAt;
  }

  _changeLanePreference(pref) {
    this.lanePreference = pref;

    if (pref > 0) {
      this.laneLeftDom.classList.add('is-outlined');
      this.laneLeftDom.classList.remove('is-selected');
      this.laneRightDom.classList.remove('is-outlined');
      this.laneRightDom.classList.add('is-selected');
    } else {
      this.laneRightDom.classList.add('is-outlined');
      this.laneRightDom.classList.remove('is-selected');
      this.laneLeftDom.classList.remove('is-outlined');
      this.laneLeftDom.classList.add('is-selected');
    }
  }

  _changeStartMode(mode) {
    this.startMode = mode == 'manual' ? 'manual' : 'autonomous';

    if (this.startMode == 'manual') {
      this.startModeAutonomousDom.classList.add('is-outlined');
      this.startModeAutonomousDom.classList.remove('is-selected');
      this.startModeManualDom.classList.remove('is-outlined');
      this.startModeManualDom.classList.add('is-selected');
    } else {
      this.startModeManualDom.classList.add('is-outlined');
      this.startModeManualDom.classList.remove('is-selected');
      this.startModeAutonomousDom.classList.remove('is-outlined');
      this.startModeAutonomousDom.classList.add('is-selected');
    }
  }

  setStartMode(mode, saveHistory = true) {
    this._changeStartMode(mode);
    if (saveHistory) this._commitHistory();
  }

  _commitHistory() {
    if (this.isRestoringHistory || this.isApplyingScenario) return;

    const snapshot = JSON.stringify(this.scenarioToJSON());
    const historyLength = this.undoHistory.length;

    if (historyLength == 0 || this.undoHistory[historyLength - 1] !== snapshot) {
      this.undoHistory.push(snapshot);

      if (this.undoHistory.length > MAX_UNDO_ENTRIES)
        this.undoHistory.shift();
    }
  }

  undo() {
    if (this.undoHistory.length <= 1) return;

    this.undoHistory.pop();
    const previousSnapshot = this.undoHistory[this.undoHistory.length - 1];
    this.isRestoringHistory = true;
    try {
      this.loadJSON(JSON.parse(previousSnapshot));
    } finally {
      this.isRestoringHistory = false;
    }
  }

  saveClicked() {
    const name = window.prompt('Name your scenario:', this.previousSavedName || '');
    if (name === null) return;
    if (name === '') {
      window.alert('The scenario name cannot be blank.');
      return;
    }

    let [success, savedAt] = this.scenarioManager.saveScenario(name, this.scenarioToJSON(), name === this.previousSavedName);
    const formattedSavedAt = formatDate(savedAt);

    if (success) {
      this.updateSavedInfo(name, formattedSavedAt);
    } else if (confirm(`A scenario named "${name}" already exists, last saved ${formattedSavedAt}. Do you want to overwrite it?`)) {
      [success, savedAt] = this.scenarioManager.saveScenario(name, this.scenarioToJSON(), true);
      this.updateSavedInfo(name, formatDate(savedAt));
    }
  }

  loadClicked() {
    this.scenarioManager.showModal();
  }

  shareClicked() {
    this.shareManager.showModal(this.scenarioToJSON());
  }

  _dimensionsFromRect(from, to) {
    const center = from.clone().add(to).divideScalar(2);
    const width = Math.max(0.5, Math.abs(from.x - to.x));
    const height = Math.max(0.5, Math.abs(from.z - to.z));
    return [center, width, height];
  }
}
