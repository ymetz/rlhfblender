export default class ManualController {
  constructor() {
    this.carKeys = { forward: false, backward: false, left: false, right: false, brake: false };
    this.remoteKeys = { forward: false, backward: false, left: false, right: false, brake: false };

    document.addEventListener('keydown', event => {
      switch (event.key) {
        case 'w': case 'W': this.carKeys.forward = true; break;
        case 's': case 'S': this.carKeys.backward = true; break;
        case 'a': case 'A': this.carKeys.left = true; break;
        case 'd': case 'D': this.carKeys.right = true; break;
        case ' ': this.carKeys.brake = true; break;
      }
    });

    document.addEventListener('keyup', event => {
      switch (event.key) {
        case 'w': case 'W': this.carKeys.forward = false; break;
        case 's': case 'S': this.carKeys.backward = false; break;
        case 'a': case 'A': this.carKeys.left = false; break;
        case 'd': case 'D': this.carKeys.right = false; break;
        case ' ': this.carKeys.brake = false; break;
      }
    });
  }

  setRemoteKeys(keys = {}) {
    this.remoteKeys = {
      forward: !!keys.forward,
      backward: !!keys.backward,
      left: !!keys.left,
      right: !!keys.right,
      brake: !!keys.brake
    };
  }

  clearRemoteKeys() {
    this.setRemoteKeys();
  }

  control() {
    let gas = 0;
    let brake = 0;
    let steer = 0;

    const keys = {
      forward: this.carKeys.forward || this.remoteKeys.forward,
      backward: this.carKeys.backward || this.remoteKeys.backward,
      left: this.carKeys.left || this.remoteKeys.left,
      right: this.carKeys.right || this.remoteKeys.right,
      brake: this.carKeys.brake || this.remoteKeys.brake
    };

    if (keys.forward) gas += 1;
    if (keys.backward) gas -= 1;
    if (keys.left) steer -= 1;
    if (keys.right) steer += 1;
    if (keys.brake) brake += 1;

    return { gas, brake, steer };
  }
}
