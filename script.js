// GridWorld Engine and Interactive Logic

const SIZE = 4;
const ACTIONS = {
    'u': {r: -1, c: 0},
    'd': {r: 1, c: 0},
    'l': {r: 0, c: -1},
    'r': {r: 0, c: 1}
};

class GridWorld {
    constructor(mode, containerId, statusId) {
        this.mode = mode; // 'static', 'player', 'random'
        this.container = document.getElementById(containerId);
        this.statusEl = document.getElementById(statusId);
        
        this.player = {r: 0, c: 0};
        this.goal = {r: 0, c: 0};
        this.pit = {r: 0, c: 0};
        this.wall = {r: 0, c: 0};
        
        this.isDone = false;
        this.initGrid();
    }

    randPair() {
        return {r: Math.floor(Math.random() * SIZE), c: Math.floor(Math.random() * SIZE)};
    }

    isEqual(pos1, pos2) {
        return pos1.r === pos2.r && pos1.c === pos2.c;
    }

    validateBoard() {
        const positions = [this.player, this.goal, this.pit, this.wall];
        for(let i=0; i<positions.length; i++){
            for(let j=i+1; j<positions.length; j++){
                if(this.isEqual(positions[i], positions[j])) return false;
            }
        }
        return true;
    }

    initGrid() {
        if (this.mode === 'static') {
            this.player = {r: 0, c: 3};
            this.goal = {r: 0, c: 0};
            this.pit = {r: 0, c: 1};
            this.wall = {r: 1, c: 1};
        } else if (this.mode === 'player') {
            this.goal = {r: 0, c: 0};
            this.pit = {r: 0, c: 1};
            this.wall = {r: 1, c: 1};
            do {
                this.player = this.randPair();
            } while (!this.validateBoard());
        } else if (this.mode === 'random') {
            do {
                this.player = this.randPair();
                this.goal = this.randPair();
                this.pit = this.randPair();
                this.wall = this.randPair();
            } while (!this.validateBoard());
        }
        
        this.isDone = false;
        this.updateStatus('環境已重置。請操作或點擊模擬按鈕。');
        this.render();
    }

    movePlayer(actionKey) {
        if(this.isDone) return;

        const move = ACTIONS[actionKey];
        const newPos = {r: this.player.r + move.r, c: this.player.c + move.c};

        // Out of bounds
        if(newPos.r < 0 || newPos.r >= SIZE || newPos.c < 0 || newPos.c >= SIZE) {
            this.updateStatus('撞到邊界！ (-1)');
            return -1;
        }

        // Wall
        if(this.isEqual(newPos, this.wall)) {
            this.updateStatus('撞到牆壁！ (-1)');
            return -1;
        }

        this.player = newPos;
        this.render();

        // Check outcomes
        if(this.isEqual(this.player, this.goal)) {
            this.isDone = true;
            this.updateStatus('到達目標！ 獲得 +10 分 ⭐');
            return 10;
        } else if(this.isEqual(this.player, this.pit)) {
            this.isDone = true;
            this.updateStatus('掉入陷阱！ 獲得 -10 分 🔥');
            return -10;
        }

        this.updateStatus('移動成功 (-1)');
        return -1;
    }

    render() {
        this.container.innerHTML = '';
        for(let r=0; r<SIZE; r++){
            for(let c=0; c<SIZE; c++){
                const cell = document.createElement('div');
                cell.className = 'cell';
                const pos = {r, c};

                if(this.isEqual(pos, this.player)) {
                    cell.classList.add('player');
                    cell.innerText = '👤';
                } else if(this.isEqual(pos, this.goal)) {
                    cell.classList.add('goal');
                    cell.innerText = '⭐';
                } else if(this.isEqual(pos, this.pit)) {
                    cell.classList.add('pit');
                    cell.innerText = '🔥';
                } else if(this.isEqual(pos, this.wall)) {
                    cell.classList.add('wall');
                    cell.innerText = '🧱';
                }

                this.container.appendChild(cell);
            }
        }
    }

    updateStatus(msg) {
        this.statusEl.innerText = '狀態: ' + msg;
    }
}

// Instances
const worlds = {
    1: new GridWorld('static', 'grid-1', 'status-1'),
    2: new GridWorld('player', 'grid-2', 'status-2'),
    3: new GridWorld('random', 'grid-3', 'status-3'),
    4: new GridWorld('random', 'grid-4', 'status-4'),
};

let activeSection = null;
let simulationInterval = null;

function setMode(sectionNum, mode) {
    if(simulationInterval) clearInterval(simulationInterval);
    activeSection = sectionNum;
    worlds[sectionNum].updateStatus('手動模式啟動。請使用方向鍵操作。');
}

function resetBoard(sectionNum) {
    if(simulationInterval) clearInterval(simulationInterval);
    worlds[sectionNum].initGrid();
}

// Simulated Agents
const simConfig = {
    'naive': {
        name: 'Naive DQN',
        behavior: (world) => {
            // Naive DQN is unstable, might get stuck or move randomly.
            // Simulate bad behavior sometimes
            if(Math.random() < 0.3) {
                return ['u','d','l','r'][Math.floor(Math.random()*4)];
            }
            return getOptimalMove(world);
        }
    },
    'replay': {
        name: 'Experience Replay',
        behavior: (world) => getOptimalMove(world) // Stable optimal move
    },
    'double': {
        name: 'Double DQN',
        behavior: (world) => getOptimalMove(world)
    },
    'dueling': {
        name: 'Dueling DQN',
        behavior: (world) => getOptimalMove(world)
    },
    'lightning': {
        name: 'Lightning DQN',
        behavior: (world) => getOptimalMove(world)
    },
    'rainbow': {
        name: 'Rainbow DQN',
        behavior: (world) => getOptimalMove(world)
    }
};

// A simple A* or BFS to find the goal to simulate a trained agent
function getOptimalMove(world) {
    // Very simple BFS to find path to goal, avoiding pit and wall
    let queue = [{pos: world.player, path: []}];
    let visited = new Set();
    visited.add(`${world.player.r},${world.player.c}`);

    while(queue.length > 0) {
        let curr = queue.shift();
        
        if(world.isEqual(curr.pos, world.goal)) {
            return curr.path[0]; // Return the first step
        }

        const keys = Object.keys(ACTIONS);
        // Shuffle keys to avoid bias
        keys.sort(() => Math.random() - 0.5);

        for(let action of keys) {
            let m = ACTIONS[action];
            let nextPos = {r: curr.pos.r + m.r, c: curr.pos.c + m.c};
            let posKey = `${nextPos.r},${nextPos.c}`;

            if(nextPos.r >= 0 && nextPos.r < SIZE && nextPos.c >= 0 && nextPos.c < SIZE) {
                if(!visited.has(posKey) && !world.isEqual(nextPos, world.wall) && !world.isEqual(nextPos, world.pit)) {
                    visited.add(posKey);
                    queue.push({pos: nextPos, path: [...curr.path, action]});
                }
            }
        }
    }
    
    // If no safe path, just do random valid move
    return ['u','d','l','r'][Math.floor(Math.random()*4)];
}

function simulateAgent(sectionNum, agentType) {
    if(simulationInterval) clearInterval(simulationInterval);
    const world = worlds[sectionNum];
    const agent = simConfig[agentType];
    
    world.initGrid(); // Reset before sim
    world.updateStatus(`模擬 ${agent.name} 運行中...`);

    simulationInterval = setInterval(() => {
        if(world.isDone) {
            clearInterval(simulationInterval);
            return;
        }
        
        const action = agent.behavior(world);
        world.movePlayer(action);
        
    }, 500); // Move every 500ms
}

// Global Keyboard Listener
window.addEventListener('keydown', (e) => {
    if(!activeSection) return;
    
    const world = worlds[activeSection];
    let action = null;
    
    switch(e.key) {
        case 'ArrowUp': action = 'u'; break;
        case 'ArrowDown': action = 'd'; break;
        case 'ArrowLeft': action = 'l'; break;
        case 'ArrowRight': action = 'r'; break;
    }

    if(action) {
        e.preventDefault();
        world.movePlayer(action);
    }
});
