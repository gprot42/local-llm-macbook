const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const scoreElement = document.getElementById('score');

canvas.width = 800;
canvas.height = 400;

const gravity = 0.5;

const TILE_SIZE = 40;
const LEVEL_MAP = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

const ENEMY_MAP = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

class Tile {
    constructor(x, y, type) {
        this.x = x;
        this.y = y;
        this.width = TILE_SIZE;
        this.height = TILE_SIZE;
        this.type = type; // 1 for ground, etc.
    }

    draw() {
        if (this.type === 1) {
            ctx.fillStyle = '#8B4513'; // Brown for ground
            ctx.fillRect(this.x, this.y, this.width, this.height);
            // Add a little detail to the tile
            ctx.strokeStyle = '#5D2E0D';
            ctx.strokeRect(this.x, this.y, this.width, this.height);
        }
    }
}

function rectIntersect(rect1, rect2) {
  return rect1.x < rect2.x + rect2.width &&
         rect1.x + rect1.width > rect2.x &&
         rect1.y < rect2.y + rect2.height &&
         rect1.y + rect1.height > rect2.y;
}

class Enemy {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.width = 30;
    this.height = 30;
    this.velocityX = 2;
    this.velocityY = 0;
    this.speed = 2;
    this.gravity = 0.6;
    this.color = 'brown';
  }

  update(tiles) {
    // Apply gravity
    this.velocityY += this.gravity;

    // Apply horizontal movement
    this.x += this.velocityX;
    this.handleXCollisions(tiles);

    // Apply vertical movement
    this.y += this.velocityY;
    this.handleYCollisions(tiles);
  }

  handleXCollisions(tiles) {
    for (let tile of tiles) {
      if (rectIntersect(this, tile)) {
        if (this.velocityX > 0) {
          this.x = tile.x - this.width;
        } else if (this.velocityX < 0) {
          this.x = tile.x + tile.width;
        }
        this.velocityX *= -1; // Bounce back
      }
    }
  }

  handleYCollisions(tiles) {
    for (let tile of tiles) {
      if (rectIntersect(this, tile)) {
        if (this.velocityY > 0) {
          this.y = tile.y - this.height;
          this.velocityY = 0;
        } else if (this.velocityY < 0) {
          this.y = tile.y + tile.height;
          this.velocityY = 0;
        }
      }
    }
  }

  draw(ctx) {
    ctx.fillStyle = this.color;
    ctx.fillRect(this.x, this.y, this.width, this.height);
  }
}


    draw() {
        ctx.fillStyle = this.color;
        ctx.fillRect(this.position.x, this.position.y, this.width, this.height);
    }

  update(tiles) {
    // Apply horizontal movement
    this.x += this.dx;
    this.handleXCollisions(tiles);

    // Apply vertical movement
    this.dy += gravity;
    this.y += this.dy;
    this.grounded = false;
    this.handleYCollisions(tiles);
  }

  handleXCollisions(tiles) {
    for (const tile of tiles) {
      if (rectIntersect(this.x, this.y, this.width, this.height, tile.x, tile.y, tile.width, tile.height)) {
        if (this.dx > 0) {
          this.x = tile.x - this.width;
        } else if (this.dx < 0) {
          this.x = tile.x + tile.width;
        }
        this.dx = 0;
      }
    }
  }

  handleYCollisions(tiles) {
    for (const tile of tiles) {
      if (rectIntersect(this.x, this.y, this.width, this.height, tile.x, tile.y, tile.width, tile.height)) {
        if (this.dy > 0) {
          this.y = tile.y - this.height;
          this.grounded = true;
        } else if (this.dy < 0) {
          this.y = tile.y + tile.height;
        }
        this.dy = 0;
      }
    }
  }
    }
}

const player = new Player();
const keys = {
    right: false,
    left: false,
    up: false
};

function animate() {
    requestAnimationFrame(animate);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    player.update();

    // Handle keyboard input
    if (keys.right) {
        player.velocity.x = player.speed;
    } else if (keys.left) {
        player.velocity.x = -player.speed;
    } else {
        player.velocity.x = 0;
    }
}

window.addEventListener('keydown', (event) => {
    switch (event.code) {
        case 'ArrowRight':
        case 'KeyD':
            keys.right = true;
            break;
        case 'ArrowLeft':
        case 'KeyA':
            keys.left = true;
            break;
        case 'ArrowUp':
        case 'Space':
        case 'KeyW':
            if (player.grounded) {
                player.velocity.y = player.jumpPower;
                player.grounded = false;
            }
            break;
    }
});

window.addEventListener('keyup', (event) => {
    switch (event.code) {
        case 'ArrowRight':
        case 'KeyD':
            keys.right = false;
            break;
        case 'ArrowLeft':
        case 'KeyA':
            keys.left = false;
            break;
    }
});

animate();
