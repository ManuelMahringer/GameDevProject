# 1. Team members

| Name                  | Matr.Nr.  | E-Mail                    |
| --------------------- | --------- | ------------------------- |
| Manuel Mahringer      | K11816360 | k11816360@students.jku.at |
| Lukas Seifriedsberger | K11816320 | k11816320@students.jku.at |
| Michael Duschek       | K11844534 | k11844534@students.jku.at |

# 2. Responsibilities
| Feature                        | Responsible                |
| ------------------------------ | -------------------------- |
| Core game world                | Mahringer                  |
| Networking                     | Mahringer, Seifriedsberger |
| Game world serialization       | Seifriedsberger            |
| Server synced countdowns       | Mahringer                  |
| Server synced sounds           | Mahringer                  |
| Player movement                | Seifriedsberger            |
| Player/world interaction       | Mahringer, Seifriedsberger |
| Player/player interaction      | Seifriedsberger            |
| Player (re)spawn               | Mahringer, Seifriedsberger |
| UI                             | Mahringer, Seifriedsberger |
| Lobby                          | Mahringer, Seifriedsberger |
| Lighting                       | Mahringer, Seifriedsberger |
| Balancing                      | Mahringer, Seifriedsberger |
| Save/load maps                 | Seifriedsberger            |
| Safe zones                     | Seifriedsberger            |
| Flag and base logic            | Seifriedsberger            |
| 100 hours code refactoring :-) | Seifriedsberger            |
| Weapon models                  | Duschek                    |
| Player model + Textures        | Duschek                    |
| Shooting + Building Animations | Duschek                    |
| Block Textures                 | Duschek                    |

# 3. Game controls
## Player
| Key                                              | Action   |
| ------------------------------------------------ | -------- |
| <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> | Movement |
| <kbd>Shift</kbd>                                 | Sprint   |
| <kbd>Space</kbd>                                 | Jump     |

## Weapons
| Key          | Action         |
| ------------ | -------------- |
| <kbd>1</kbd> | Assault Rifle  |
| <kbd>2</kbd> | Pistol         |
| <kbd>3</kbd> | Shovel + Block |

When using weapon 1 (Assault Rifle) or weapon 2 (Pistol): <kbd>Left click</kbd> to **shoot**   
When using weapon 3 (Shovel + Block): 
* <kbd>Left click</kbd> to **gather a block** ressource or hit player (melee range)
* <kbd>Right click</kbd> to **build a block** at the highlighted destination
* <kbd>Mouse wheel up/down</kbd> to **choose block type** to build

## General
| Key            | Action      |
| -------------- | ----------- |
| <kbd>Esc</kbd> | Ingame menu |


# 4. Instructions to run the game
The game features 2 different modes:   
**Host** - A game instances started as host acts as server and client simulatiously.    
**Client** - A game instance started as client can only send and request data to/from the server.  
Each client action is processed on the server (= host).  

## Local testing
To test the game locally using multiple instances please follow these steps:  

1. Start a game instance, press play and choose "HOST". 
Enter your name, choose a team via buttons "JOIN BLUE" or "JOIN RED" and a map using the "SELECT MAP" dropdown menu.  

2. Start up to 5 additional game instances, press play and choose "CLIENT" on each of them. 
Enter your name and chose a team for each of the clients.  

3. When all clients are connected the host may start start via the "START GAME" button". (The amount of connected players is visible in the lobby UI)

Maps have to be located in a folder called "Maps", that is located in the same directory as directory the `Guerras De Cubos.exe`, in order for the game to find them. Every map itself is stored in its own folder. 
For example, we have a map called "Default", so it is stored like the following: `./Maps/Default/`. The `Default` folder would then contain all the chunk files of the map.

*Attention:* The IP adress of the Host as well as the adress of the clients to connect to is set to 127.0.0.1 to unconditionally support local testing.  
Therefore playtesting using distributed clients is currently not possible.  
Our interal playtesting as well the recording of the video was done using a VPN, in our case Hamachi (https://www.vpn.net/).

The IP Adress can be changed in the Unity Editor -> Menu (Scene) -> NetworkManager (GameObject) -> U Net Transport (Script) -> Connect Address

## Running in Unity
When running the game in Unity it is important that the game is started from the  `Menu` scene. Otherwise components will not get initialized properly.

Maps are again located in a folder called "Maps" at the root directory of the project solution.

# 5. Assets

Attack Jump & Hit Damage Human Sounds
https://assetstore.unity.com/packages/audio/sound-fx/voices/attack-jump-hit-damage-human-sounds-32785

Footsteps - Essentials
https://assetstore.unity.com/packages/audio/sound-fx/foley/footsteps-essentials-189879

AllSky Free - 10 Sky / Skybox Set
https://assetstore.unity.com/packages/2d/textures-materials/sky/allsky-free-10-sky-skybox-set-146014

Free Pixel Font - Thaleah
https://assetstore.unity.com/packages/2d/fonts/free-pixel-font-thaleah-140059\

Mixkit - Free Sound Effects 
https://mixkit.co/free-sound-effects/

Blades & Bludgeoning Free Sample Pack
https://assetstore.unity.com/packages/audio/sound-fx/blades-bludgeoning-free-sample-pack-179306

Bullet Imact Sound: https://freesound.org/people/morganpurkis/sounds/392975/\

# 6. Building maps
We built in a secondary player mode called "Build", that supports flying, ignores world collision, has unlimited building/mining range and an (practically) unlimited amount of blocks. Additionally, it supports saving the current map state and loading a new map during the game. We used this gamemode to build and serialize our map.

## Build mode - additional controls
| Key              | Action                      |
| ---------------- | --------------------------- |
| <kbd>Z</kbd>     | Save/serialize map popup    |
| <kbd>U</kbd>     | Load/deserialize map poupup |
| <kbd>Space</kbd> | Fly up                      |
| <kbd>Shift</kbd> | Fly down                    |

Maps are saved in the root directory of the project in the folder "Maps".

To activate the building mode, the game mode dropdown has to be enabled in the Unity Editor -> Menu (Scene) -> Canvas -> MainMenu -> Main Menu (Script) -> Enable Game Mode Dropdown.

When activated and the "Play" button has been pressed, a dropdown should show up, where the user can choose between "Build" mode and "Fight" mode (the default game mode).

_Disclaimer:_ The building mode is not very polished, as it is just used internally for creating the maps and is not intended to be released as a separate "game mode" as of now.

# 7. Game description/goal
Guerras de cubos  is a 2-team first-person capture-the-flag (ctf) shooter in a voxel world. The world can be modified by building/destroying blocks, excluding safe-zones (team bases, flag spawn). Teams consist of 1 up to 3 players per team.

The goal of the game is to be the first team to manage to capture the flag 10 times. A flag is captured by picking it up, and carrying it back to the drop-off location own team base. The drop-off location is indicated by a cube in team-color, that is located inside the base building. Picking up and dropping the flag just requires colliding with the flag/the drop-off location. The flag itself is located at a platform in the sky, so building is required to get to the flag.

The players may interfere with the goal of the other team getting the flag by attacking them with their weapons. On death, players are penalized by a respawn timer and losing all items in the inventory.

# 8. Game world core idea
The idea of game world based on voxels stems from the following thread:
https://forum.unity.com/threads/coredev-creating-voxelised-worlds-like-minecraft.192954/


