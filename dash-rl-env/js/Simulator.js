import Physics from "./physics/Physics.js";
import Path from "./autonomy/Path.js";
import CubicPath from "./autonomy/path-planning/CubicPath.js";
import FollowController from "./autonomy/control/FollowController.js";
import ManualController from "./autonomy/control/ManualController.js";
import MapObject from "./objects/MapObject.js";
import CarObject from "./objects/CarObject.js";
import StaticObstacleObject from "./objects/StaticObstacleObject.js";
import DynamicObstacleObject from "./objects/DynamicObstacleObject.js";
import Editor from "./simulator/Editor.js";
import OrbitControls from "./simulator/OrbitControls.js";
import TopDownCameraControls from "./simulator/TopDownCameraControls.js";
import Dashboard from "./simulator/Dashboard.js";
import EXAMPLES from "./simulator/examples.js";
import RoadLattice from "./autonomy/path-planning/RoadLattice.js";
import StaticObstacle from "./autonomy/StaticObstacle.js";
import MovingAverage from "./autonomy/MovingAverage.js";
import PathPlannerConfigEditor from "./simulator/PathPlannerConfigEditor.js";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import Car from "./physics/Car.js";

const WELCOME_MODAL_KEY = 'dash_WelcomeModal';
const BRIDGE_PROTOCOL_VERSION = '1.0';
const RL_DEFAULT_DT = 1 / 60;
const RL_DEFAULT_ACTION_REPEAT = 6;
const RL_DEFAULT_MAX_STEPS = 2000;

export default class Simulator {
  constructor(domElement) {
    this.pathPlannerWorker = new Worker(new URL('../workers/PathPlannerWorker.js', import.meta.url), { type: 'module' });
    this.pathPlannerWorker.onmessage = this.receivePlannedPath.bind(this);
    this.pathPlannerConfigEditor = new PathPlannerConfigEditor();

    this.physics = new Physics();
    this.car = this.physics.createCar();

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(domElement.clientWidth, domElement.clientHeight);
    this.renderer.shadowMap.enabled = true;
    domElement.appendChild(this.renderer.domElement);

    this.lastPlanParams = null;
    this.renderer.getContext().canvas.addEventListener('webglcontextlost', event => {
      console.log('Simulator: webgl context lost');
      console.log(event);
      console.log(this.lastPlanParams);
    });

    this._setUpCameras(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.sceneFog = null;//new THREE.FogExp2(0x111111, 0.0025);
    this.scene.fog = this.sceneFog;
    this.scene.background = new THREE.Color(0x111111);

    this.editor = new Editor(this.renderer.domElement, this.editorCamera, this.scene);

    const geolocation = null;//[33.523900, -111.908756];
    const map = new MapObject(geolocation);
    this.scene.add(map);

    this.carObject = new CarObject(this.car);
    this.scene.add(this.carObject);

    this.scene.add(new THREE.AmbientLight(0x666666));
    const light = new THREE.DirectionalLight(0xffffff, 0.75);
    light.position.set(1, 1, 1).normalize();
    this.scene.add(light);

    this.manualCarController = new ManualController();
    this.autonomousCarController = null;

    this.dashboard = new Dashboard(this.car);

    this.plannerReady = false;
    this.plannerRunning = false;
    this.plannerReset = false;
    this.carStation = null;
    this.plannedPathGroup = new THREE.Group();
    this.scene.add(this.plannedPathGroup);

    this.staticObstaclesGroup = new THREE.Group();
    this.scene.add(this.staticObstaclesGroup);
    this.dynamicObstaclesGroup = new THREE.Group();
    this.scene.add(this.dynamicObstaclesGroup);

    this.paused = false;
    this.prevTimestamp = null;
    this.frameCounter = 0;
    this.fpsTime = 0;
    this.fps = 0;
    this.simulatedTime = 0;
    this.lastPlanTime = null;
    this.averagePlanTime = new MovingAverage(20);
    this.lastAppliedControls = { steer: 0, brake: 0, gas: 0 };

    this.rlConfig = {
      dt: RL_DEFAULT_DT,
      actionRepeat: RL_DEFAULT_ACTION_REPEAT,
      maxSteps: RL_DEFAULT_MAX_STEPS,
      lanePenaltyWeight: 0.02,
      steerPenaltyWeight: 0.01,
      brakePenaltyWeight: 0.005,
      stepPenalty: 0.01,
      collisionPenalty: 10,
      offroadPenalty: 5
    };
    this.rlEpisodeStep = 0;
    this.rlLastStation = 0;
    this.rlLastSteer = 0;

    this.trajectoryRecorder = {
      enabled: false,
      frames: []
    };
    this.humanRecorder = {
      enabled: false,
      requireManualMode: true,
      sampleInterval: 0.1,
      elapsed: 0,
      lastStation: 0,
      lastSteer: 0,
      frames: []
    };
    this.loadedRecording = null;
    this.bridgeHostOrigin = null;
    this.bridgeAllowedOrigins = new Set();

    window.addEventListener('resize', e => {
      this._updateCameraAspects(domElement.clientWidth / domElement.clientHeight);
      this.renderer.setSize(domElement.clientWidth, domElement.clientHeight);
    });

    window.addEventListener('hashchange', e => {
      if (window.location.hash.startsWith('#/s/'))
        window.location.reload();
    });

    this.manualModeButton = document.getElementById('mode-manual');
    this.manualModeButton.addEventListener('click', this.enableManualMode.bind(this));
    this.autonomousModeButton = document.getElementById('mode-autonomous');
    this.autonomousModeButton.addEventListener('click', this.enableAutonomousMode.bind(this));

    document.getElementById('editor-enable').addEventListener('click', this.enableEditor.bind(this));
    document.getElementById('editor-finalize').addEventListener('click', this.finalizeEditor.bind(this));
    document.getElementById('simulator-load').addEventListener('click', this.loadScenario.bind(this));

    this.scenarioPlayButton = document.getElementById('scenario-play');
    this.scenarioPlayButton.addEventListener('click', this.playScenario.bind(this));
    this.scenarioPauseButton = document.getElementById('scenario-pause');
    this.scenarioPauseButton.addEventListener('click', this.pauseScenario.bind(this));
    this.scenarioRestartButton = document.getElementById('scenario-restart');
    this.scenarioRestartButton.addEventListener('click', this.restartScenario.bind(this));
    this.teleopRecordStartButton = document.getElementById('teleop-record-start');
    this.teleopRecordStopButton = document.getElementById('teleop-record-stop');
    this.teleopRecordCountDom = document.getElementById('teleop-record-count');
    this.teleopRecordStartButton.addEventListener('click', () => this.startHumanTeleopRecording({ sampleHz: 10, clear: true, requireManualMode: true }));
    this.teleopRecordStopButton.addEventListener('click', () => this.stopHumanTeleopRecording());
    this._updateHumanRecorderUi();

    this.welcomeModal = document.getElementById('welcome-modal');
    document.getElementById('show-welcome-modal').addEventListener('click', e => this.welcomeModal.classList.add('is-active'));

    this.uiMode = this._resolveUiMode();
    this._applyUiMode();

    if (window.localStorage.getItem(WELCOME_MODAL_KEY) !== 'hide') {
      this.welcomeModal.classList.add('is-active');
    }

    document.getElementById('welcome-modal-background').addEventListener('click', this.hideWelcomeModal.bind(this));
    document.getElementById('welcome-modal-close').addEventListener('click', this.hideWelcomeModal.bind(this));

    document.getElementById('welcome-modal-examples').addEventListener('click', e => {
      this.welcomeModal.classList.remove('is-active');
      this.loadScenario();
      this.editor.scenarioManager.switchTab(this.editor.scenarioManager.examplesTab);
    });

    document.getElementById('welcome-modal-create').addEventListener('click', e => {
      this.welcomeModal.classList.remove('is-active');
      this.enableEditor();
    });

    this.simModeBoxes = Array.prototype.slice.call(document.getElementsByClassName('sim-mode-box'), 0);
    this.editModeBoxes = Array.prototype.slice.call(document.getElementsByClassName('edit-mode-box'), 0);

    this.fpsBox = document.getElementById('fps');

    this.enableManualMode();
    this.changeCamera('chase');

    this.aroundAnchorIndex = null;
    this.staticObstacles = [];
    this.dynamicObstacles = [];

    this._checkHashScenario();
    this._setUpMessageBridge();

    requestAnimationFrame(this.step.bind(this));
  }

  _resolveUiMode() {
    const search = new URLSearchParams(window.location.search);
    const requested = (search.get('ui') || '').trim().toLowerCase();

    if (requested === 'minimal' || requested === 'external')
      return 'minimal';

    if (requested === 'gym' || requested === 'recording')
      return 'gym';

    return 'default';
  }

  _applyUiMode() {
    document.body.classList.remove('ui-minimal', 'ui-gym');

    const container = document.getElementById('container');
    const dashboard = document.getElementById('dashboard');
    if (!container || !dashboard) return;

    let panel = document.getElementById('render-info-panel');
    if (this.uiMode === 'gym') {
      if (!panel) {
        panel = document.createElement('div');
        panel.id = 'render-info-panel';
        document.body.appendChild(panel);
      }

      if (dashboard.parentNode !== panel)
        panel.appendChild(dashboard);

      document.body.classList.add('ui-gym');
      window.localStorage.setItem(WELCOME_MODAL_KEY, 'hide');
      if (this.welcomeModal)
        this.welcomeModal.classList.remove('is-active');
      return;
    }

    if (panel && dashboard.parentNode !== container)
      container.insertBefore(dashboard, container.firstChild);

    if (panel)
      panel.remove();

    if (this.uiMode === 'minimal') {
      document.body.classList.add('ui-minimal');
      window.localStorage.setItem(WELCOME_MODAL_KEY, 'hide');
      if (this.welcomeModal)
        this.welcomeModal.classList.remove('is-active');
    }
  }

  toss() {
    const pose = this.car.pose;
    const rotVec = THREE.Vector2.fromAngle(pose.rot);
    const pos = rotVec.clone().multiplyScalar(50).add(new THREE.Vector2(rotVec.y, rotVec.x)).add(pose.pos);
    const obstacle = new StaticObstacle(pos, 0, 1.0, 1.0);

    const obsGeom = new THREE.PlaneGeometry(obstacle.width, obstacle.height);
    const obsMat = new THREE.MeshBasicMaterial({ color: 0x0000ff, depthTest: false, transparent: true, opacity: 0.5 });
    const obsObj = new THREE.Mesh(obsGeom, obsMat);
    obsObj.rotation.x = -Math.PI / 2;
    obsObj.rotation.z = -obstacle.rot;
    obsObj.position.set(obstacle.pos.x, 0, obstacle.pos.y);
    this.scene.add(obsObj);

    this.staticObstacles.push(obstacle);
  }

  _checkHashScenario() {
    const search = new URLSearchParams(window.location.search);
    const queryCode = search.get('scenario') || search.get('s') || search.get('config');
    const queryExample = search.get('scenarioName') || search.get('example');

    if (queryCode && this._importScenarioCode(queryCode)) {
      const url = new URL(window.location.href);
      url.searchParams.delete('scenario');
      url.searchParams.delete('s');
      url.searchParams.delete('config');
      url.searchParams.delete('scenarioName');
      url.searchParams.delete('example');
      window.history.replaceState({}, '', url.toString());
      return;
    }

    if (queryExample && this._importExampleScenario(queryExample)) {
      const url = new URL(window.location.href);
      url.searchParams.delete('scenarioName');
      url.searchParams.delete('example');
      window.history.replaceState({}, '', url.toString());
      return;
    }

    if (!window.location.hash.startsWith('#/s/')) return;

    const [_hash, _s, code] = window.location.hash.split('/');
    if (this._importScenarioCode(code))
      window.location.hash = '';
  }

  _importScenarioCode(encoded) {
    try {
      // Support regular base64, URL-safe base64, and URL-encoded values.
      let normalized = decodeURIComponent(encoded).replace(/-/g, '+').replace(/_/g, '/');
      const padding = normalized.length % 4;
      if (padding > 0)
        normalized += '='.repeat(4 - padding);

      const json = JSON.parse(atob(normalized));
      this.editor.loadJSON(json);
      this.finalizeEditor();
      this.welcomeModal.classList.remove('is-active');
      return true;
    } catch (e) {
      console.log('Error importing scenario code:');
      console.log(encoded);
      console.log(e);
      return false;
    }
  }

  _normalizeScenarioName(name) {
    if (typeof(name) !== 'string') return '';
    return name
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9\s_-]+/g, '')
      .replace(/[\s-]+/g, '_')
      .replace(/_+/g, '_');
  }

  _cloneScenarioData(scenarioData) {
    if (typeof structuredClone === 'function')
      return structuredClone(scenarioData);

    return JSON.parse(JSON.stringify(scenarioData));
  }

  _findExampleScenarioByName(name) {
    const normalized = this._normalizeScenarioName(name);
    if (!normalized) return null;

    const compact = normalized.replace(/_/g, '');
    for (const example of EXAMPLES) {
      const exampleNormalized = this._normalizeScenarioName(example.name);
      if (normalized === exampleNormalized || compact === exampleNormalized.replace(/_/g, ''))
        return example;
    }

    return null;
  }

  _importExampleScenario(name) {
    const example = this._findExampleScenarioByName(name);
    if (!example) return false;

    this.editor.loadJSON(this._cloneScenarioData(example.data));
    this.finalizeEditor(false);
    this.welcomeModal.classList.remove('is-active');
    return true;
  }

  _setUpMessageBridge() {
    const search = new URLSearchParams(window.location.search);
    const hostOrigin = search.get('hostOrigin');
    this.bridgeHostOrigin = hostOrigin || null;

    if (window.location.ancestorOrigins && window.location.ancestorOrigins.length > 0) {
      for (let i = 0; i < window.location.ancestorOrigins.length; i++)
        this.bridgeAllowedOrigins.add(window.location.ancestorOrigins[i]);
    }

    if (hostOrigin)
      this.bridgeAllowedOrigins.add(hostOrigin);

    window.addEventListener('message', this._handleBridgeMessage.bind(this));

    this._bridgePost('dash.ready', {
      capabilities: [
        'dash.init',
        'dash.loadScenario',
        'dash.loadRecording',
        'dash.play',
        'dash.pause',
        'dash.reset',
        'dash.getState',
        'dash.setMode',
        'dash.setKeys',
        'dash.clearKeys',
        'dash.seek'
      ]
    });
  }

  _bridgeStatePayload() {
    const observation = this._buildObservation();
    return {
      simTime: observation.time,
      paused: this.paused,
      mode: this.carControllerMode,
      car: {
        x: this.car.position.x,
        y: this.car.position.y,
        rot: this.car.rotation,
        speed: this.car.velocity
      },
      station: observation.station,
      latitude: observation.latitude
    };
  }

  _captureStateSnapshot() {
    return {
      simTime: this.simulatedTime,
      paused: this.paused,
      mode: this.carControllerMode,
      car: {
        x: this.car.position.x,
        y: this.car.position.y,
        rot: this.car.rotation,
        velocity: this.car.velocity,
        acceleration: this.car.acceleration,
        wheelAngle: this.car.wheelAngle,
        wheelAngularVelocity: this.car.wheelAngularVelocity
      },
      carStation: this.carStation,
      aroundAnchorIndex: this.aroundAnchorIndex
    };
  }

  _applyStateSnapshot(snapshot) {
    if (!snapshot || !snapshot.car) throw new Error('Invalid state snapshot.');

    this.simulatedTime = Number(snapshot.simTime) || 0;
    this.car.position.set(Number(snapshot.car.x) || 0, Number(snapshot.car.y) || 0);
    this.car.rotation = Number(snapshot.car.rot) || 0;
    this.car.velocity = Number(snapshot.car.velocity) || 0;
    this.car.acceleration = Number(snapshot.car.acceleration) || 0;
    this.car.wheelAngle = Number(snapshot.car.wheelAngle) || 0;
    this.car.wheelAngularVelocity = Number(snapshot.car.wheelAngularVelocity) || 0;
    this.carStation = typeof(snapshot.carStation) === 'number' ? snapshot.carStation : 0;
    this.aroundAnchorIndex = typeof(snapshot.aroundAnchorIndex) === 'number' ? snapshot.aroundAnchorIndex : null;

    if (snapshot.mode === 'autonomous')
      this.enableAutonomousMode();
    else
      this.enableManualMode();

    if (snapshot.paused)
      this.pauseScenario();
    else
      this.playScenario();

    this.updateDynamicObjects(this.simulatedTime);
    this._resetChaseCamera();
    this._resetFreeCamera();
    this._resetTopDownCamera();
  }

  _loadRecording(recording) {
    if (!recording || !Array.isArray(recording.frames))
      throw new Error('Recording payload must include a frames array.');

    this.loadedRecording = recording;
  }

  _seekRecordingByFrame(frameIndex) {
    if (!this.loadedRecording || !Array.isArray(this.loadedRecording.frames))
      throw new Error('No recording is loaded.');

    const frames = this.loadedRecording.frames;
    const idx = Math.max(0, Math.min(frames.length - 1, Math.floor(frameIndex)));
    const frame = frames[idx];
    if (!frame) throw new Error(`Frame ${idx} does not exist.`);

    if (frame.stateSnapshot) {
      this._applyStateSnapshot(frame.stateSnapshot);
      return { frameIndex: idx, simTime: this.simulatedTime };
    }

    throw new Error('Recording frame has no state snapshot.');
  }

  _seekRecordingByTime(timeSeconds) {
    if (!this.loadedRecording || !Array.isArray(this.loadedRecording.frames))
      throw new Error('No recording is loaded.');

    const target = Number(timeSeconds);
    if (!Number.isFinite(target))
      throw new Error('Seek time must be a finite number.');

    let bestIndex = -1;
    let bestDelta = Number.POSITIVE_INFINITY;
    this.loadedRecording.frames.forEach((frame, index) => {
      const t = Number(frame.t);
      if (!Number.isFinite(t)) return;
      const delta = Math.abs(t - target);
      if (delta < bestDelta) {
        bestDelta = delta;
        bestIndex = index;
      }
    });

    if (bestIndex < 0)
      throw new Error('Recording does not contain valid time stamps.');

    return this._seekRecordingByFrame(bestIndex);
  }

  _bridgePost(type, payload, id = undefined, error = undefined) {
    const targetOrigin = this.bridgeHostOrigin || '*';
    if (!window.parent || window.parent === window) return;

    window.parent.postMessage({
      source: 'dash-player',
      version: BRIDGE_PROTOCOL_VERSION,
      type,
      id,
      payload,
      error
    }, targetOrigin);
  }

  _bridgeAck(id, command, data = undefined) {
    this._bridgePost('dash.ack', { command, ok: true, data }, id);
  }

  _bridgeNack(id, command, code, message, details = undefined) {
    this._bridgePost('dash.nack', { command, ok: false }, id, { code, message, details });
  }

  _applyBridgeScenario(payload = {}) {
    if (payload.scenarioJson) {
      this.editor.loadJSON(this._cloneScenarioData(payload.scenarioJson));
      this.finalizeEditor(false);
    } else if (payload.scenarioCode) {
      if (!this._importScenarioCode(payload.scenarioCode))
        throw new Error('Invalid scenario code.');
    } else if (payload.scenarioName) {
      if (!this._importExampleScenario(payload.scenarioName))
        throw new Error(`Unknown example scenario "${payload.scenarioName}".`);
    }

    if (payload.startMode === 'autonomous')
      this.enableAutonomousMode();
    else if (payload.startMode === 'manual')
      this.enableManualMode();

    if (payload.paused === true)
      this.pauseScenario();
    else if (payload.paused === false)
      this.playScenario();

    if (payload.recording)
      this._loadRecording(payload.recording);
  }

  _handleBridgeMessage(event) {
    if (this.bridgeAllowedOrigins.size > 0 && !this.bridgeAllowedOrigins.has(event.origin))
      return;

    const message = event.data || {};
    if (message.source !== 'dash-host' || message.version !== BRIDGE_PROTOCOL_VERSION || typeof(message.type) !== 'string')
      return;

    const id = message.id;
    const type = message.type;
    const payload = message.payload || {};

    try {
      switch (type) {
        case 'dash.init':
        case 'dash.loadScenario':
          this._applyBridgeScenario(payload);
          this._bridgeAck(id, type);
          this._bridgePost('dash.event', { name: 'loaded' });
          break;
        case 'dash.loadRecording':
          this._loadRecording(payload.recording || payload);
          this._bridgeAck(id, type, { frames: this.loadedRecording.frames.length });
          break;
        case 'dash.play':
          this.playScenario();
          this._bridgeAck(id, type);
          this._bridgePost('dash.event', { name: 'started' });
          break;
        case 'dash.pause':
          this.pauseScenario();
          this._bridgeAck(id, type);
          this._bridgePost('dash.event', { name: 'paused' });
          break;
        case 'dash.reset':
          this.restartScenario();
          this._bridgeAck(id, type);
          break;
        case 'dash.getState':
          this._bridgePost('dash.state', this._bridgeStatePayload(), id);
          break;
        case 'dash.setMode':
          if (payload.mode === 'autonomous')
            this.enableAutonomousMode();
          else
            this.enableManualMode();
          this._bridgeAck(id, type);
          break;
        case 'dash.setKeys':
          this.manualCarController.setRemoteKeys(payload);
          this.enableManualMode();
          this._bridgeAck(id, type);
          break;
        case 'dash.clearKeys':
          this.manualCarController.clearRemoteKeys();
          this._bridgeAck(id, type);
          break;
        case 'dash.seek':
          if (payload.mode === 'time') {
            const result = this._seekRecordingByTime(payload.value);
            this._bridgePost('dash.frame', { ...result, state: this._bridgeStatePayload() }, id);
          } else if (payload.mode === 'frame') {
            const result = this._seekRecordingByFrame(payload.value);
            this._bridgePost('dash.frame', { ...result, state: this._bridgeStatePayload() }, id);
          } else {
            this._bridgeNack(id, type, 'BAD_REQUEST', 'Seek mode must be "time" or "frame".');
          }
          break;
        default:
          this._bridgeNack(id, type, 'UNSUPPORTED_COMMAND', `Unknown command "${type}".`);
      }
    } catch (error) {
      this._bridgeNack(
        id,
        type,
        'INTERNAL_ERROR',
        error && error.message ? error.message : 'Unknown bridge error.',
        { stack: error && error.stack ? error.stack : undefined }
      );
    }
  }

  _setUpCameras(domElement) {
    this.chaseCamera = new THREE.PerspectiveCamera(55, domElement.clientWidth / domElement.clientHeight, 1, 10000);
    this.chaseCameraControls = new OrbitControls(this.chaseCamera, domElement);
    this.chaseCameraControls.minDistance = 4;
    this.chaseCameraControls.maxDistance = 5000;
    this.chaseCameraControls.maxPolarAngle = Math.PI / 2.02;
    this.chaseCameraControls.enablePan = false;
    this.chaseCameraControls.enabled = false;
    this._resetChaseCamera();

    this.freeCamera = new THREE.PerspectiveCamera(55, domElement.clientWidth / domElement.clientHeight, 1, 10000);
    this.freeCameraControls = new OrbitControls(this.freeCamera, domElement);
    this.freeCameraControls.minDistance = 5;
    this.freeCameraControls.maxDistance = 5000;
    this.freeCameraControls.maxPolarAngle = Math.PI / 2.02;
    this.freeCameraControls.enabled = true;
    this._resetFreeCamera();

    this.topDownCamera = new THREE.PerspectiveCamera(55, domElement.clientWidth / domElement.clientHeight, 1, 10000);
    this.topDownCamera.position.set(0, 50, 0);
    this.topDownCamera.lookAt(0, 0, 0);
    this.topDownControls = new TopDownCameraControls(domElement, this.topDownCamera);
    this.topDownControls.enabled = false;
    this.topDownControls.minAltitude = 5;
    this.topDownControls.maxAltitude = 10000;

    this.editorCamera = new THREE.PerspectiveCamera(45, domElement.clientWidth / domElement.clientHeight, 1, 10000);
    this.editorCamera.layers.enable(2);
    this.editorCamera.position.set(0, 200, 0);
    this.editorCamera.lookAt(0, 0, 0);
    this.editorCameraControls = new TopDownCameraControls(domElement, this.editorCamera);
    this.editorCameraControls.enabled = false;
    this.editorCameraControls.enablePanning = true;
    this.editorCameraControls.minAltitude = 10;
    this.editorCameraControls.maxAltitude = 10000;

    this.cameraButtons = {};

    ['free', 'chase', 'topDown'].forEach(c => {
      const cameraButton = document.getElementById(`camera-${c}`);
      cameraButton.addEventListener('click', () => this.changeCamera(c));
      this.cameraButtons[c] = cameraButton;
    });

    this.switchTo2DButton = document.getElementById('camera-2D');
    this.switchTo2DButton.addEventListener('click', this.switchTo2D.bind(this));
    this.switchTo3DButton = document.getElementById('camera-3D');
    this.switchTo3DButton.addEventListener('click', this.switchTo3D.bind(this));

    this.switchTo3D();
  }

  _resetFreeCamera() {
    this.freeCameraControls.position0.copy(this.chaseCamera.position);
    const carPosition = this.car.position;
    this.freeCameraControls.target0.set(carPosition.x, 0, carPosition.y);
    this.freeCameraControls.reset();
  }

  _resetChaseCamera() {
    const pos = this.car.position;
    this.chaseCamera.up.set(0, 1, 0);

    const dirVector = THREE.Vector2.fromAngle(this.car.rotation).multiplyScalar(-20);
    this.chaseCamera.position.set(pos.x + dirVector.x, 8, pos.y + dirVector.y);
    this.chaseCameraControls.target.set(pos.x, 0, pos.y);
    this.chaseCamera.lookAt(pos.x, 0, pos.y);

    // Synchronize OrbitControls baseline to this exact pose so load/reset
    // cannot reuse stale orbit radius/angles from previous camera state.
    this.chaseCameraControls.saveState();
    this.chaseCameraControls.reset();
    this.chaseCameraControls.update();
  }

  _resetTopDownCamera() {
    const carPosition = this.car.position;
    this.topDownCamera.position.set(carPosition.x, 50, carPosition.y);
    this.topDownCamera.rotation.z = -this.car.rotation - Math.PI / 2
  }

  _updateCameraAspects(aspect) {
    this.freeCamera.aspect = aspect;
    this.freeCamera.updateProjectionMatrix();
    this.chaseCamera.aspect = aspect;
    this.chaseCamera.updateProjectionMatrix();
    this.topDownCamera.aspect = aspect;
    this.topDownCamera.updateProjectionMatrix();
    this.editorCamera.aspect = aspect;
    this.editorCamera.updateProjectionMatrix();
  }

  enableEditor() {
    this.editor.enabled = true;
    this.editor.setStartMode(this.carControllerMode || 'autonomous', false);
    this.plannerRunning = false;

    this.previousCamera = this.camera;
    this.camera = this.editorCamera;
    this.editorCameraControls.enabled = true;
    this.chaseCameraControls.enabled = false;
    this.topDownControls.enabled = false;
    this.freeCameraControls.enabled = false;

    this.scene.fog = null;
    this.carObject.visible = false;
    if (this.plannedPathGroup) this.plannedPathGroup.visible = false;
    this.staticObstaclesGroup.visible = false;
    this.dynamicObstaclesGroup.visible = false;

    this.simModeBoxes.forEach(el => el.classList.add('is-hidden'));
    this.editModeBoxes.forEach(el => el.classList.remove('is-hidden'));
  }

  finalizeEditor(replaceCamera = true) {
    this.editor.enabled = false;
    this.editorCameraControls.enabled = false;

    this.scene.fog = this.sceneFog;
    this.carObject.visible = true;

    this.simModeBoxes.forEach(el => el.classList.remove('is-hidden'));
    this.editModeBoxes.forEach(el => el.classList.add('is-hidden'));

    if (this.editor.lanePath.anchors.length > 1) {
      const centerline = this.editor.lanePath.centerline;
      const pos = centerline[0].clone();
      const dir = centerline[1].clone().sub(centerline[0]);
      const rot = Math.atan2(dir.y, dir.x);
      const perpindicular = rot + Math.PI / 2 * (Math.sign(this.editor.lanePreference) || 0);
      const latitude = this.pathPlannerConfigEditor.config.roadWidth / 4;

      this.car.setPose(pos.x + Math.cos(perpindicular) * latitude, pos.y + Math.sin(perpindicular) * latitude, rot);
      this.car.velocity = this.editor.initialSpeed;

      this.dynamicObstacles = this.editor.dynamicObstacles;

      // The `false` value means the controller is waiting to be created after the first planning cycle.
      // This signals the simulator to use neutral controls instead of the hard brake used for the `null` value.
      this.autonomousCarController = false;
      const startsAutonomous = this.editor.startMode !== 'manual';
      if (startsAutonomous)
        this.enableAutonomousMode();
      else
        this.enableManualMode();

      if (!this.plannerRunning) {
        this.plannerReady = true;
        this.plannerRunning = true;
      }
      this.plannerReset = true;
      this.simulatedTime = 0;
      this.carStation = 0;
      this.aroundAnchorIndex = null;

      if (startsAutonomous) {
        this.pauseScenario();
        this.autonomousModeButton.classList.add('is-loading');
        this.waitingForFirstPlan = true;
      } else {
        this.playScenario();
        this.autonomousModeButton.classList.remove('is-loading');
        this.waitingForFirstPlan = false;
      }
    } else {
      this.dynamicObstacles = [];
    }

    this.staticObstacles = this.editor.staticObstacles;
    this.recreateStaticObstacleObjects();
    this.recreateDynamicObstacleObjects();

    this.dashboard.update({ steer: 0, brake: 0, gas: 0 }, this.car.velocity, null, null, 0, this.averagePlanTime.average);

    if (replaceCamera) {
      this.camera = this.previousCamera;

      if (this.previousCamera == this.chaseCamera)
        this.chaseCameraControls.enabled = true;
      else if (this.previousCamera == this.topDownCamera)
        this.topDownControls.enabled = true;
      else if (this.previousCamera == this.freeCamera)
        this.freeCameraControls.enabled = true;
      else
        this.changeCamera('chase');
    }

    this._resetChaseCamera();
    this._resetFreeCamera();
    this._resetTopDownCamera();
  }

  recreateStaticObstacleObjects() {
    this.scene.remove(this.staticObstaclesGroup);
    this.staticObstaclesGroup = new THREE.Group();
    this.scene.add(this.staticObstaclesGroup);

    this.staticObstacles.forEach(o => {
      const obstacleObject = new StaticObstacleObject(o);
      this.staticObstaclesGroup.add(obstacleObject);
    });
  }

  recreateDynamicObstacleObjects() {
    this.scene.remove(this.dynamicObstaclesGroup);
    this.dynamicObstaclesGroup = new THREE.Group();
    this.scene.add(this.dynamicObstaclesGroup);

    this.dynamicObstacles.forEach(o => {
      const obstacleObject = new DynamicObstacleObject(o, this.editor.lanePath);
      this.dynamicObstaclesGroup.add(obstacleObject);
    });

    this.updateDynamicObjects(this.simulatedTime);
  }

  updateDynamicObjects(time) {
    this.dynamicObstaclesGroup.children.forEach(o => o.update(time));
  }

  playScenario() {
    this.paused = false;
    this.scenarioPlayButton.classList.add('is-hidden');
    this.scenarioPauseButton.classList.remove('is-hidden');
  }

  pauseScenario() {
    this.paused = true;
    this.scenarioPlayButton.classList.remove('is-hidden');
    this.scenarioPauseButton.classList.add('is-hidden');
  }

  restartScenario() {
    if (this.editor.enabled) return;

    if (this.plannedPathGroup)
      this.scene.remove(this.plannedPathGroup);

    this.finalizeEditor(false);
  }

  loadScenario() {
    if (this.editor.enabled) return;

    this.editor.scenarioManager.showModal(this.finalizeEditor.bind(this));
  }

  enableManualMode() {
    this.manualModeButton.classList.remove('is-outlined');
    this.manualModeButton.classList.add('is-selected');
    this.autonomousModeButton.classList.add('is-outlined', 'is-inverted');
    this.autonomousModeButton.classList.remove('is-selected', 'is-link');

    this.carControllerMode = 'manual';
  }

  enableAutonomousMode() {
    this.autonomousModeButton.classList.remove('is-outlined', 'is-inverted');
    this.autonomousModeButton.classList.add('is-selected', 'is-link');
    this.manualModeButton.classList.add('is-outlined');
    this.manualModeButton.classList.remove('is-selected');

    this.carControllerMode = 'autonomous';
  }

  changeCamera(mode) {
    if (this.editor.enabled) return;

    switch (mode) {
      case "free":
        this.chaseCameraControls.enabled = false;
        this.topDownControls.enabled = false;
        this.freeCameraControls.enabled = true;

        if (this.camera == this.freeCamera)
          this._resetFreeCamera();
        else
          this.camera = this.freeCamera;

        break;
      case "chase":
        this.freeCameraControls.enabled = false;
        this.topDownControls.enabled = false;
        this.chaseCameraControls.enabled = true;

        if (this.camera == this.chaseCamera)
          this._resetChaseCamera();
        else
          this.camera = this.chaseCamera;

        break;
      case "topDown":
        this.freeCameraControls.enabled = false;
        this.chaseCameraControls.enabled = false;
        this.topDownControls.enabled = true;

        if (this.camera == this.topDownCamera)
          this._resetTopDownCamera();
        else
          this.camera = this.topDownCamera;

        break;
      default:
        console.log(`Unknown camera mode: ${mode}`);
        return;
    }

    for (const c in this.cameraButtons) {
      const classes = this.cameraButtons[c].classList;
      if (c == mode) {
        classes.remove('is-outlined');
        classes.add('is-selected');
      } else {
        classes.add('is-outlined');
        classes.remove('is-selected');
      }
    }
  }

  switchTo2D() {
    this.switchTo2DButton.classList.remove('is-outlined');
    this.switchTo2DButton.classList.add('is-selected');
    this.switchTo3DButton.classList.add('is-outlined');
    this.switchTo3DButton.classList.remove('is-selected');

    this.chaseCamera.layers.enable(2);
    this.topDownCamera.layers.enable(2);
    this.freeCamera.layers.enable(2);
    this.chaseCamera.layers.disable(3);
    this.topDownCamera.layers.disable(3);
    this.freeCamera.layers.disable(3);
  }

  switchTo3D() {
    this.switchTo3DButton.classList.remove('is-outlined');
    this.switchTo3DButton.classList.add('is-selected');
    this.switchTo2DButton.classList.add('is-outlined');
    this.switchTo2DButton.classList.remove('is-selected');

    this.chaseCamera.layers.enable(3);
    this.topDownCamera.layers.enable(3);
    this.freeCamera.layers.enable(3);
    this.chaseCamera.layers.disable(2);
    this.topDownCamera.layers.disable(2);
    this.freeCamera.layers.disable(2);
  }

  _normalizeAction(action = {}) {
    return {
      steer: Math.clamp(Number(action.steer) || 0, -1, 1),
      gas: Math.clamp(Number(action.gas) || 0, 0, 1),
      brake: Math.clamp(Number(action.brake) || 0, 0, 1)
    };
  }

  _finiteNumber(value, fallback = 0, clampMagnitude = 1e6) {
    const n = Number(value);
    if (!Number.isFinite(n))
      return fallback;

    if (clampMagnitude > 0)
      return Math.clamp(n, -clampMagnitude, clampMagnitude);

    return n;
  }

  _laneInfoForPose() {
    if (this.editor.lanePath.anchors.length <= 1)
      return null;

    const carRearAxle = this.car.rearAxlePosition;
    const [station, latitude, aroundAnchorIndex] = this.editor.lanePath.stationLatitudeFromPosition(carRearAxle, this.aroundAnchorIndex);
    if (Number.isFinite(aroundAnchorIndex))
      this.aroundAnchorIndex = aroundAnchorIndex;

    const safeStation = this._finiteNumber(station, this._finiteNumber(this.carStation, 0, 1e7), 1e7);
    const safeLatitude = this._finiteNumber(latitude, 0, 1e5);
    const [sample] = this.editor.lanePath.sampleStations(safeStation, 1, 0);
    const sampleRot = sample ? this._finiteNumber(sample.rot, this.car.rotation, Math.PI * 8) : this.car.rotation;
    const headingError = this._finiteNumber(Math.wrapAngle(this.car.rotation - sampleRot), 0, Math.PI * 2);
    const curvature = sample ? this._finiteNumber(sample.curv, 0, 100) : 0;

    this.carStation = safeStation;
    return { station: safeStation, latitude: safeLatitude, headingError, curvature };
  }

  _dynamicObstaclePose(dynamicObstacle, time) {
    const slPos = dynamicObstacle.positionAtTime(time);
    const [sample] = this.editor.lanePath.sampleStations(slPos.x, 1, 0);
    if (!sample) return null;

    const rot = sample.rot;
    const pos = THREE.Vector2.fromAngle(rot + Math.PI / 2).multiplyScalar(slPos.y).add(sample.pos);
    return { x: pos.x, y: pos.y, rot };
  }

  _intersectsExpandedOrientedRect(point, rectCenter, rectRot, halfWidth, halfHeight, expandRadius) {
    const dx = point.x - rectCenter.x;
    const dy = point.y - rectCenter.y;
    const cos = Math.cos(rectRot);
    const sin = Math.sin(rectRot);

    const localX = dx * cos + dy * sin;
    const localY = -dx * sin + dy * cos;

    return Math.abs(localX) <= halfWidth + expandRadius && Math.abs(localY) <= halfHeight + expandRadius;
  }

  _hasCollision() {
    const carCenter = this.car.position;
    const carRadius = Math.hypot(Car.HALF_CAR_LENGTH, Car.HALF_CAR_WIDTH) * 0.5;

    for (const obstacle of this.staticObstacles) {
      if (this._intersectsExpandedOrientedRect(
        carCenter,
        obstacle.pos,
        obstacle.rot,
        obstacle.width / 2,
        obstacle.height / 2,
        carRadius
      )) {
        return true;
      }
    }

    for (const obstacle of this.dynamicObstacles) {
      const pose = this._dynamicObstaclePose(obstacle, this.simulatedTime);
      if (!pose) continue;

      if (this._intersectsExpandedOrientedRect(
        carCenter,
        { x: pose.x, y: pose.y },
        pose.rot,
        obstacle.size.w,
        obstacle.size.h,
        carRadius
      )) {
        return true;
      }
    }

    return false;
  }

  _nearestObstacleDistance() {
    const carCenter = this.car.position;
    let best = Number.POSITIVE_INFINITY;

    this.staticObstacles.forEach(o => {
      const d = carCenter.distanceTo(o.pos);
      if (d < best) best = d;
    });

    this.dynamicObstacles.forEach(o => {
      const pose = this._dynamicObstaclePose(o, this.simulatedTime);
      if (!pose) return;
      const d = Math.hypot(carCenter.x - pose.x, carCenter.y - pose.y);
      if (d < best) best = d;
    });

    return Number.isFinite(best) ? best : null;
  }

  _buildObservation() {
    const laneInfo = this._laneInfoForPose();
    const roadHalfWidth = this._finiteNumber(this.pathPlannerConfigEditor.config.roadWidth / 2, 10, 1e4);
    const latitude = laneInfo ? this._finiteNumber(laneInfo.latitude, 0, 1e5) : 0;
    const offroad = !!(laneInfo && Math.abs(latitude) > roadHalfWidth);
    const nearestObstacleDistance = this._finiteNumber(this._nearestObstacleDistance(), 1e6, 1e6);

    return {
      time: this._finiteNumber(this.simulatedTime, 0, 1e7),
      speed: this._finiteNumber(this.car.velocity, 0, 1e4),
      steering: this._finiteNumber(this.car.wheelAngle, 0, Math.PI),
      station: laneInfo ? this._finiteNumber(laneInfo.station, 0, 1e7) : 0,
      latitude: latitude,
      headingError: laneInfo ? this._finiteNumber(laneInfo.headingError, 0, Math.PI) : 0,
      curvature: laneInfo ? this._finiteNumber(laneInfo.curvature, 0, 100) : 0,
      nearestObstacleDistance: nearestObstacleDistance,
      offroad: offroad,
      collision: !!this._hasCollision()
    };
  }

  _evaluateTransition(previousStation, controls, previousSteer = this.rlLastSteer) {
    const laneInfo = this._laneInfoForPose();
    const roadHalfWidth = this._finiteNumber(this.pathPlannerConfigEditor.config.roadWidth / 2, 10, 1e4);

    const safePreviousStation = this._finiteNumber(previousStation, this._finiteNumber(this.rlLastStation, 0, 1e7), 1e7);
    const station = laneInfo ? this._finiteNumber(laneInfo.station, safePreviousStation, 1e7) : safePreviousStation;
    const latitude = laneInfo ? this._finiteNumber(laneInfo.latitude, 0, 1e5) : 0;
    const progressReward = station - safePreviousStation;
    const lanePenaltyWeight = Number(this.rlConfig.lanePenaltyWeight) || 0;
    const steerPenaltyWeight = Number(this.rlConfig.steerPenaltyWeight) || 0;
    const brakePenaltyWeight = Number(this.rlConfig.brakePenaltyWeight) || 0;
    const offroadPenaltyWeight = Number(this.rlConfig.offroadPenalty) || 0;
    const collisionPenaltyWeight = Number(this.rlConfig.collisionPenalty) || 0;
    const stepPenalty = Number(this.rlConfig.stepPenalty) || 0;
    const lanePenalty = Math.abs(latitude) * lanePenaltyWeight;
    const steerPenalty = Math.abs(controls.steer - this._finiteNumber(previousSteer, 0, Math.PI)) * steerPenaltyWeight;
    const brakePenalty = controls.brake * brakePenaltyWeight;

    const offroad = laneInfo ? Math.abs(latitude) > roadHalfWidth : false;
    const collision = this._hasCollision();

    let reward = progressReward - lanePenalty - steerPenalty - brakePenalty - stepPenalty;
    if (offroad) reward -= offroadPenaltyWeight;
    if (collision) reward -= collisionPenaltyWeight;
    reward = this._finiteNumber(reward, 0, 1e6);

    return {
      station,
      offroad,
      collision,
      terminated: offroad || collision,
      reward,
      nextSteer: controls.steer,
      components: {
        progress: progressReward,
        lanePenalty: lanePenalty,
        steerPenalty: steerPenalty,
        brakePenalty: brakePenalty,
        stepPenalty: stepPenalty,
        offroadPenalty: offroad ? offroadPenaltyWeight : 0,
        collisionPenalty: collision ? collisionPenaltyWeight : 0
      }
    };
  }

  _simulateTick(dt, controlOverride = null, force = false) {
    if (this.editor.enabled) return null;
    if (!force && this.paused) return null;

    this.simulatedTime += dt;

    const prevCarPosition = this.car.position;
    const prevCarRotation = this.car.rotation;

    const manualControls = this.manualCarController.control(this.car.pose, this.car.wheelAngle, this.car.velocity, dt);
    if (controlOverride === null && (manualControls.steer != 0 || manualControls.brake != 0 || manualControls.gas != 0))
      this.enableManualMode();

    let autonomousControls = { steer: 0, brake: 0, gas: 0 };
    if (this.autonomousCarController)
      autonomousControls = this.autonomousCarController.control(this.car.pose, this.car.wheelAngle, this.car.velocity, dt, this.carControllerMode == 'autonomous');
    else if (this.autonomousCarController === null)
      autonomousControls = { steer: 0, brake: 1, gas: 0 };

    const controls = controlOverride || (this.carControllerMode == 'autonomous' ? autonomousControls : manualControls);

    this.car.update(controls, dt);
    this.physics.step(dt);
    this.updateDynamicObjects(this.simulatedTime);

    const carPosition = this.car.position;
    const carRotation = this.car.rotation;

    const positionOffset = { x: carPosition.x - prevCarPosition.x, y: 0, z: carPosition.y - prevCarPosition.y };
    this.chaseCamera.position.add(positionOffset);
    this.chaseCameraControls.target.set(carPosition.x, 0, carPosition.y);
    this.chaseCameraControls.rotateLeft(carRotation - prevCarRotation);
    this.chaseCameraControls.update();

    this.topDownCamera.position.setX(carPosition.x);
    this.topDownCamera.position.setZ(carPosition.y);
    this.topDownCamera.rotation.z = -carRotation - Math.PI / 2;

    const laneInfo = this._laneInfoForPose();
    this.dashboard.update(controls, this.car.velocity, this.carStation, laneInfo ? laneInfo.latitude : null, this.simulatedTime, this.averagePlanTime.average);

    this.lastAppliedControls = { steer: controls.steer, brake: controls.brake, gas: controls.gas };

    if (this.trajectoryRecorder.enabled) {
      this.trajectoryRecorder.frames.push({
        t: this.simulatedTime,
        action: { ...this.lastAppliedControls },
        observation: this._buildObservation()
      });
    }

    if (this.humanRecorder.enabled)
      this._sampleHumanTeleop(dt, this.lastAppliedControls);

    return { controls: this.lastAppliedControls };
  }

  _sampleHumanTeleop(dt, controls) {
    if (this.humanRecorder.requireManualMode && this.carControllerMode !== 'manual')
      return;

    this.humanRecorder.elapsed += dt;
    while (this.humanRecorder.elapsed >= this.humanRecorder.sampleInterval) {
      this.humanRecorder.elapsed -= this.humanRecorder.sampleInterval;

      const transition = this._evaluateTransition(this.humanRecorder.lastStation, controls, this.humanRecorder.lastSteer);
      this.humanRecorder.lastStation = transition.station;
      this.humanRecorder.lastSteer = transition.nextSteer;

      this.humanRecorder.frames.push({
        frameIndex: this.humanRecorder.frames.length,
        t: this.simulatedTime,
        dt: this.humanRecorder.sampleInterval,
        action: { ...controls },
        observation: this._buildObservation(),
        stateSnapshot: this._captureStateSnapshot(),
        reward: transition.reward,
        terminated: transition.terminated,
        rewardComponents: transition.components
      });

      if (transition.terminated) {
        this.humanRecorder.enabled = false;
        this._updateHumanRecorderUi();
        break;
      }
    }

    this._updateHumanRecorderUi();
  }

  _updateHumanRecorderUi() {
    if (!this.teleopRecordStartButton || !this.teleopRecordStopButton || !this.teleopRecordCountDom)
      return;

    const sampleCount = this.humanRecorder.frames.length;
    this.teleopRecordCountDom.textContent = `${sampleCount} sample${sampleCount === 1 ? '' : 's'}`;

    if (this.humanRecorder.enabled) {
      this.teleopRecordStartButton.classList.add('is-hidden');
      this.teleopRecordStopButton.classList.remove('is-hidden');
    } else {
      this.teleopRecordStopButton.classList.add('is-hidden');
      this.teleopRecordStartButton.classList.remove('is-hidden');
    }
  }

  setRLConfig(config = {}) {
    Object.assign(this.rlConfig, config || {});
    return { ...this.rlConfig };
  }

  envReset(options = {}) {
    if (options.scenario) {
      this.editor.loadJSON(this._cloneScenarioData(options.scenario));
      this.finalizeEditor(false);
    } else if (options.scenarioCode) {
      if (!this._importScenarioCode(options.scenarioCode))
        throw new Error('Invalid scenario code.');
    } else if (options.scenarioName) {
      if (!this._importExampleScenario(options.scenarioName))
        throw new Error(`Unknown example scenario "${options.scenarioName}".`);
    } else if (this.editor.enabled) {
      this.finalizeEditor(false);
    }

    if (options.startMode == 'autonomous')
      this.enableAutonomousMode();
    else
      this.enableManualMode();

    this.pauseScenario();
    this.rlEpisodeStep = 0;
    this.rlLastSteer = 0;
    const laneInfo = this._laneInfoForPose();
    this.rlLastStation = laneInfo ? this._finiteNumber(laneInfo.station, 0, 1e7) : 0;

    if (options.clearRecording === true)
      this.clearTrajectoryRecording();

    return {
      observation: this._buildObservation(),
      info: { mode: this.carControllerMode }
    };
  }

  envStep(action, options = {}) {
    const dt = Number(options.dt) > 0 ? Number(options.dt) : this.rlConfig.dt;
    const repeat = Number(options.actionRepeat) > 0 ? Math.floor(options.actionRepeat) : this.rlConfig.actionRepeat;
    const controls = this._normalizeAction(action);

    let reward = 0;
    let terminated = false;
    let truncated = false;
    let lastTransition = null;

    for (let i = 0; i < repeat; i++) {
      const previousStation = this.rlLastStation;
      this._simulateTick(dt, controls, true);
      lastTransition = this._evaluateTransition(previousStation, controls);
      this.rlLastStation = this._finiteNumber(lastTransition.station, this.rlLastStation, 1e7);
      this.rlLastSteer = lastTransition.nextSteer;
      reward += this._finiteNumber(lastTransition.reward, 0, 1e6);
      this.rlEpisodeStep += 1;

      if (lastTransition.terminated) {
        terminated = true;
        break;
      }

      if (this.rlEpisodeStep >= this.rlConfig.maxSteps) {
        truncated = true;
        break;
      }
    }

    const observation = this._buildObservation();
    const info = {
      step: this.rlEpisodeStep,
      controls: { ...controls },
      rewardComponents: lastTransition ? lastTransition.components : {}
    };

    return { observation, reward: this._finiteNumber(reward, 0, 1e6), terminated, truncated, info };
  }

  startTrajectoryRecording({ clear = true } = {}) {
    if (clear) this.clearTrajectoryRecording();
    this.trajectoryRecorder.enabled = true;
  }

  stopTrajectoryRecording() {
    this.trajectoryRecorder.enabled = false;
    return this.getTrajectoryRecording();
  }

  clearTrajectoryRecording() {
    this.trajectoryRecorder.frames = [];
  }

  getTrajectoryRecording() {
    return this.trajectoryRecorder.frames.slice();
  }

  startHumanTeleopRecording({ sampleHz = 10, clear = true, requireManualMode = true } = {}) {
    const laneInfo = this._laneInfoForPose();

    if (clear) this.clearHumanTeleopRecording();

    this.humanRecorder.enabled = true;
    this.humanRecorder.requireManualMode = !!requireManualMode;
    this.humanRecorder.sampleInterval = 1 / Math.max(1, Number(sampleHz) || 10);
    this.humanRecorder.elapsed = 0;
    this.humanRecorder.lastStation = laneInfo ? laneInfo.station : 0;
    this.humanRecorder.lastSteer = this.lastAppliedControls.steer || 0;

    if (this.humanRecorder.requireManualMode)
      this.enableManualMode();

    this._updateHumanRecorderUi();

    return {
      sampleHz: 1 / this.humanRecorder.sampleInterval,
      requireManualMode: this.humanRecorder.requireManualMode
    };
  }

  stopHumanTeleopRecording() {
    this.humanRecorder.enabled = false;
    this._updateHumanRecorderUi();
    return this.getHumanTeleopRecording();
  }

  clearHumanTeleopRecording() {
    this.humanRecorder.frames = [];
    this._updateHumanRecorderUi();
  }

  getHumanTeleopRecording() {
    return {
      metadata: {
        sampleHz: 1 / this.humanRecorder.sampleInterval,
        requireManualMode: this.humanRecorder.requireManualMode,
        scenario: this.editor.scenarioToJSON()
      },
      frames: this.humanRecorder.frames.slice()
    };
  }

  hideWelcomeModal() {
    this.welcomeModal.classList.remove('is-active');
    window.localStorage.setItem(WELCOME_MODAL_KEY, 'hide');
  }

  startPlanner(pose, station) {
    this.plannerReady = false;
    this.lastPlanTime = performance.now();

    // In order to create a stable trajectory between successive planning
    // cycles, we must compensate for the latency between when a planning cycle
    // starts and when it ends. The average planning time is used to forward
    // simulate the vehicle to the pose it is expected to have when the
    // planning actually finishes.

    let predictedPose = pose;
    let predictedStation = station;
    let startTime = this.simulatedTime;

    if (!this.plannerReset && !this.paused && this.autonomousCarController && this.carControllerMode == 'autonomous') {
      const latency = this.averagePlanTime.average;
      predictedPose = this.autonomousCarController.predictPoseAfterTime(pose, latency);
      let [predictedStation] = this.editor.lanePath.stationLatitudeFromPosition(predictedPose.pos, this.aroundAnchorIndex);
      startTime += latency;
    }

    const reset = this.plannerReset;
    this.plannerReset = false;

    this.lastPlanParams =  {
      config: Object.assign({}, this.pathPlannerConfigEditor.config, { speedLimit: this.editor.speedLimit, lanePreference: this.editor.lanePreference }),
      vehiclePose: predictedPose,
      vehicleStation: predictedStation,
      lanePath: this.editor.lanePath,
      startTime: startTime,
      staticObstacles: this.staticObstacles,
      dynamicObstacles: this.dynamicObstacles.filter(o => o.positionAtTime(startTime).x >= 0),
      reset: reset
    };

    this.pathPlannerWorker.postMessage(this.lastPlanParams);
  }

  receivePlannedPath(event) {
    if (event.data.error) {
      document.getElementById('planner-error').classList.remove('is-hidden');
      return;
    }

    if (this.waitingForFirstPlan && !this.plannerReset) {
      this.waitingForFirstPlan = false;
      this.autonomousModeButton.classList.remove('is-loading');
      this.playScenario();
    }

    if (this.editor.enabled) return;

    const { fromVehicleParams, vehiclePose, vehicleStation, latticeStartStation, config, dynamicObstacleGrid } = event.data;
    let { path, fromVehicleSegment } = event.data;

    this.averagePlanTime.addSample((performance.now() - this.lastPlanTime) / 1000);
    this.plannerReady = true;

    if (this.plannerReset) return;

    if (this.plannedPathGroup)
      this.scene.remove(this.plannedPathGroup);
    this.plannedPathGroup = new THREE.Group();
    this.scene.add(this.plannedPathGroup);

    const circleGeom = new THREE.CircleGeometry(0.1, 32);
    const circleMat = new THREE.MeshBasicMaterial({ color: 0x00ff80, transparent: true, opacity: 0.7 });

    const lattice = new RoadLattice(this.editor.lanePath, latticeStartStation, config);
    lattice.lattice.forEach(cells => {
      cells.forEach(c => {
        const circle = new THREE.Mesh(circleGeom, circleMat);
        circle.position.set(c.pos.x, 0, c.pos.y);
        circle.rotation.x = -Math.PI / 2;
        this.plannedPathGroup.add(circle);
      });
    });

    // TODO: clear this up or just remove it
    if (false && dynamicObstacleGrid) {
      const dynamicGridTex = new THREE.DataTexture(dynamicObstacleGrid.data, dynamicObstacleGrid.width, dynamicObstacleGrid.height, THREE.RGBAFormat, THREE.FloatType);
      dynamicGridTex.flipY = true;
      dynamicGridTex.needsUpdate = true;

      const [gridStart] = this.editor.lanePath.sampleStations(vehicleStation, 1, 0);
      if (gridStart) {
        const dynamicGridGeom = new THREE.PlaneGeometry(dynamicObstacleGrid.width * config.slGridCellSize, dynamicObstacleGrid.height * config.slGridCellSize);
        const dynamicGridMat = new THREE.MeshBasicMaterial({ map: dynamicGridTex, depthTest: false, transparent: true, opacity: 0.5 });
        const dynamicGridObj = new THREE.Mesh(dynamicGridGeom, dynamicGridMat);
        dynamicGridObj.rotation.x = -Math.PI / 2;
        dynamicGridObj.rotation.z = -gridStart.rot;
        const offset = THREE.Vector2.fromAngle(gridStart.rot).multiplyScalar(dynamicObstacleGrid.width * config.slGridCellSize / 2 - config.spatialHorizon / config.lattice.numStations);
        dynamicGridObj.position.set(gridStart.pos.x + offset.x, 0, gridStart.pos.y + offset.y);

        this.plannedPathGroup.add(dynamicGridObj);
      }
    }

    if (path === null) {
      this.autonomousCarController = null;
      return;
    }

    if (fromVehicleParams.type == 'cubic') {
      const start = this.car.pose;
      const end = fromVehicleSegment[fromVehicleSegment.length - 1];

      const pathBuilder = new CubicPath(start, end, fromVehicleParams.params);

      if (pathBuilder.optimize()) {
        fromVehicleSegment = pathBuilder.buildPath(Math.ceil(pathBuilder.params.sG / 0.25));

        const prevVelocitySq = this.car.velocity * this.car.velocity;
        const accel = (end.velocity * end.velocity - prevVelocitySq) / 2 / pathBuilder.params.sG;
        const ds = pathBuilder.params.sG / (fromVehicleSegment.length - 1);
        let s = 0;

        for (let p = 0; p < fromVehicleSegment.length; p++) {
          fromVehicleSegment[p].velocity = Math.sqrt(2 * accel * s + prevVelocitySq);
          fromVehicleSegment[p].acceleration = accel;
          s += ds;
        }
      }
    }

    path = fromVehicleSegment.concat(path);

    path.forEach(p => Object.setPrototypeOf(p.pos, THREE.Vector2.prototype));
    const followPath = new Path(path);

    if (this.autonomousCarController)
      this.autonomousCarController.replacePath(followPath);
    else
      this.autonomousCarController = new FollowController(followPath, this.car);

    const linePositions = new Array(path.length * 3);
    for (let i = 0; i < path.length; i++) {
      linePositions[i * 3] = path[i].pos.x;
      linePositions[i * 3 + 1] = 0;
      linePositions[i * 3 + 2] = path[i].pos.y;
    }
    const pathGeometry = new LineGeometry();
    pathGeometry.setPositions(linePositions);

    const color = fromVehicleParams.type == 'cubic' ? new THREE.Color(0xff8800) : new THREE.Color(0xffff40);
    const pathObject = new Line2(
      pathGeometry,
      new LineMaterial({
        color: color,
        linewidth: 0.15,
        worldUnits: true,
        resolution: new THREE.Vector2(this.renderer.domElement.clientWidth, this.renderer.domElement.clientHeight)
      })
    );
    pathObject.renderOrder = 1;
    this.plannedPathGroup.add(pathObject);
  }

  step(timestamp) {
    if (this.prevTimestamp == null) {
      this.prevTimestamp = timestamp;
      requestAnimationFrame(this.step.bind(this));
      return;
    }

    const dt = (timestamp - this.prevTimestamp) / 1000;

    this.editor.update();

    this._simulateTick(dt);

    if (!this.editor.enabled && this.plannerReady) {
      this.startPlanner(this.car.pose, this.carStation || 0);
      this.dashboard.updatePlanTime(this.averagePlanTime.average);
    }

    this.frameCounter++;
    this.fpsTime += dt;
    if (this.fpsTime >= 1) {
      this.fps = this.frameCounter / this.fpsTime;
      this.frameCounter = 0;
      this.fpsTime = 0;
      this.fpsBox.textContent = this.fps.toFixed(1);
    }

    this.renderer.render(this.scene, this.camera);

    this.prevTimestamp = timestamp;

    requestAnimationFrame(this.step.bind(this));
  }
}
