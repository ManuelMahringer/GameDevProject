# 1 Team members

|Name| Matr.Nr. | E-Mail|
|---|---|---|
| Manuel Mahringer		| K11816360 | k11816360@students.jku.at
| Lukas Seifriedsberger ||  
| Michael Duschek		||


# 2 Responsibilities
|Feature   | Responsible   |
|---|---|
| Core game world       | Mahringer  
| Networking 				    | Mahringer, Seifriedsberger  
| Game world serialization	| Seifriedsberger  
| Server synced sound 		| Mahringer  
| Player movement			| Seifriedsberger  
| Player/world interaction 	| Mahringer, Seifriedsberger  
| Player/player interaction | Seifriedsberger  
| Player (re)spawn			| Mahringer, Seifriedsberger  
| Server synced countdown	| Mahringer  
| UI						| Mahringer, Seifriedsberger  
| Lobby 					| Mahringer, Seifriedsberger  
| Lighting 					| Mahringer, Seifriedsberger  
| Balancing 				| Mahringer, Seifriedsberger  
| Safe zones 				| Seifriedsberger  
| Flag and base logic		| Seifriedsberger  
| 1000 hours code refactoring | Seifriedsberger  
| Weapon models				| Duschek  
| Shooting Animations		| Duschek  

# 3 Game controls
Movement: WASD  
Sprint: Hold Shift  
Jump: Space bar  

Choose weapon: 1, 2, 3  
1: Pistol  
2: Assault Rifle  
3: Shovel  

When using weapon 1 (Pistol) or weapon 2(Assault rifle): Left click to shoot   
When using weapon 3 (Shovel): Left click to gather a block ressource or hit player (melee range), Right click to build a block at the highlighted destination

# 4 Instructions to run the game
The game features 2 different  modes: 
Host - A game instances started as Host acts as Server and Client simulatiously. 
Client - A game instance started as client can only send and request data to/from the server.
Each client action is processed on the server (= host)

To test the game locally using multiple instances please follow these steps:
Start a game instance, press play and choose "HOST".
Enter your name, choose a team via buttons "JOIN BLUE" and "JOIN RED" and map via the "SELECT MAP" dropdown menu.

Start up to 5 additional game instances, press play and choose "CLIENT" on each of them.
Enter your name and chose a team.

When all clients are connected the host may start start via the "START GAME" button".


Attention: The IP adress of the Host as well as the adress of the clients to connect to currently is 127.0.0.1 to unconditionally support local testing.
Therefore playtesting using distributed clients is currently not possible.
Our interal playtesting as well the recording of the video was done using a VPN, in our case Hamachi (https://www.vpn.net/).  

# 5 Assets
Attack Jump & Hit Damage Human Sounds 
https://assetstore.unity.com/packages/audio/sound-fx/voices/attack-jump-hit-damage-human-sounds-32785

Footsteps - Essentials
https://assetstore.unity.com/packages/audio/sound-fx/foley/footsteps-essentials-189879

AllSky Free - 10 Sky / Skybox Set
https://assetstore.unity.com/packages/2d/textures-materials/sky/allsky-free-10-sky-skybox-set-146014

Free Pixel Font - Thaleah
https://assetstore.unity.com/packages/2d/fonts/free-pixel-font-thaleah-140059\



# 6 Game world core idea
The idea of game world based on voxels stems from the following thread:
https://forum.unity.com/threads/coredev-creating-voxelised-worlds-like-minecraft.192954/

