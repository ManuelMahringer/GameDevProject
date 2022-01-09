using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Security;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Serialization;
using UnityEngine.Networking;
using Unity.Netcode;
using UnityEditor;
using UnityEngine.Networking.Types;

// using UnityEditor.Build.Content;
// using UnityEngine.Networking.Types;

public enum GameMode {
    Build,
    Fight
}

// Player Controller Basics: https://www.youtube.com/watch?v=NEUzB5vPYrE
// Fall Damage: https://www.youtube.com/watch?v=D897sarRL3w
// Health Bar:
//  - https://www.youtube.com/watch?v=BLfNP4Sc_iA
//  - https://www.youtube.com/watch?v=Gtw7VyuMdDc
public class Player : NetworkBehaviour {
    private static readonly string WorldTag = "World";
    private static readonly string PlayerTag = "Player";
    private static readonly string WeaponLayerName = "Weapon";

    [Header("Health & Damage")] public float maxHealth = 100f;

    public float fallDmgDistThreshold = 2.5f;
    public float fallDmgMultiplier = 15f;

    [Header("Walk / Run Settings")] public float walkSpeed = 3.5f;

    public float runSpeed = 5f;

    [Header("Jump Settings")] public float jumpForce = 12_000f;

    public ForceMode appliedForceMode = ForceMode.Force;

    [Header("Build/Destroy Settings")] public float hitRange = 5f;

    [Header("Ground Tag Specification")] public String groundTag = "";

    [Header("Jumping State")] [SerializeField]
    private bool jump;

    [SerializeField] private bool isGrounded;

    [Header("Current Player Speed")] [SerializeField]
    private float currentSpeed;

    public bool popupActive;
    public bool mouseActive;

    public Lobby.Team team;

    private bool _hasFlag;
    
    private Weapon _activeWeapon;
    private BlockType _activeBlock;

    [SerializeField] private LayerMask mask;

    [SerializeField] public List<GameObject> weaponModels;

    [SerializeField] private GameObject flag;

    private readonly Dictionary<WeaponType, Weapon> weapons = new Dictionary<WeaponType, Weapon>
        {{WeaponType.Handgun, new Handgun()}, {WeaponType.AssaultRifle, new AssaultRifle()}, {WeaponType.Shovel, new Shovel()}};

    private bool IsFalling => !isGrounded && _rb.velocity.y < 0;

    private const float SensitivityHor = 5.0f;
    private const float SensitivityVer = 5.0f;
    private const float MINVert = -90.0f;
    private const float MAXVert = 90.0f;

    private AudioSync _audioSync;

    // [SerializeField]
    // private Slider healthBar;
    [SerializeField] public Slider floatingHealthBar;

    private World _world;
    private GameMode _gameMode;
    private Camera _playerCamera;
    private Camera _weaponCamera;
    private float _health;
    private Rigidbody _rb;
    private Slider _healthBar;
    private AudioSource _audioSource;
    private AudioClip _fallSound;
    private AudioClip _handgunSound;
    private AudioClip _assaultRifleSound;
    private RaycastHit _hit;
    private float _xAxis;
    private float _zAxis;
    private Vector3 _dxz;
    private Vector3 _groundLocation;
    private bool _respawn;
    private bool _wasGrounded;
    private bool _wasFalling;
    private float _startOfFall;
    private GameObject _highlightBlock;
    private MapPopup _saveMapPopup;
    private MapPopup _loadMapPopup;
    private float _rotX;
    private float _tFired;
    private PlayerInventory _inventory;
    private GameObject _flagImage;

    private enum RaycastAction {
        DestroyBlock,
        BuildBlock,
        Shoot,
        HighlightBlock
    }
    
    private void Start() {
        GameNetworkManager.RegisterPlayer(NetworkObject.NetworkObjectId, this, Lobby.Team.Blue);
        floatingHealthBar.maxValue = maxHealth;
        floatingHealthBar.value = maxHealth;

        if (!IsLocalPlayer)
            return;
        floatingHealthBar.maxValue = maxHealth;
        floatingHealthBar.value = maxHealth;
        _audioSync = GetComponent<AudioSync>();
        _gameMode = ComponentManager.gameMode;
        _world = GameObject.Find("World").GetComponent<World>();
        _rb = GetComponent<Rigidbody>();
        _audioSource = GetComponent<AudioSource>();
        _fallSound = Resources.Load("Sounds/hurt_fall") as AudioClip;
        _handgunSound = Resources.Load("Sounds/handgun") as AudioClip;
        _assaultRifleSound = Resources.Load("Sounds/assault_rifle") as AudioClip;
        _healthBar = GameObject.Find("HealthBar").GetComponent<Slider>();
        _health = maxHealth;
        _healthBar.maxValue = _health;
        _healthBar.value = _health;
        _healthBar.interactable = false;
        _healthBar.gameObject.SetActive(false);
        _highlightBlock = GameObject.Find("Highlight Slab");
        _saveMapPopup = gameObject.AddComponent<MapPopup>();
        _saveMapPopup.action = MapPopup.MapPopupAction.Save;
        _loadMapPopup = gameObject.AddComponent<MapPopup>();
        _loadMapPopup.action = MapPopup.MapPopupAction.Load;
        _activeBlock = BlockType.Grass;
        _inventory = gameObject.AddComponent<PlayerInventory>();
        _flagImage = GameObject.Find("FlagImage");
        _flagImage.SetActive(false);
        
        floatingHealthBar.gameObject.SetActive(false);
        foreach (var model in weaponModels) {
            model.layer = LayerMask.NameToLayer(WeaponLayerName);
            if (model.transform.name == WeaponType.Shovel.ToString()) {
                model.transform.GetChild(0).gameObject.layer = LayerMask.NameToLayer(WeaponLayerName);
                model.transform.GetChild(1).gameObject.layer = LayerMask.NameToLayer(WeaponLayerName);
            }
        }

        Camera[] cameras = gameObject.GetComponentsInChildren<Camera>();
        _playerCamera = cameras[0];
        _playerCamera.enabled = true;
        _weaponCamera = cameras[1];
        _weaponCamera.enabled = true;
        gameObject.GetComponentInChildren<AudioListener>().enabled = true;

        if (_gameMode == GameMode.Build) {
            _rb.isKinematic = true;
            hitRange = Single.PositiveInfinity;
            isGrounded = true;
            _healthBar.gameObject.SetActive(false);
            runSpeed = walkSpeed = 8f;
            for (int i = 0; i < _inventory.Size; i++) {
                _inventory.Items[i] = Int32.MaxValue / 2;
            }
        }

        DeactivateMouse();
        transform.position = new Vector3(0, 1, 0);

        _world.gameStarted.OnValueChanged += OnGameStarted;
        // Debug.Log("BEFORE CHUNK SEND");
        // _world.GetInitialChunkDataServerRpc();
    }
    
    private void OnGameStarted(bool oldVal, bool newVal) {
        ActivateMouse();
        SwitchWeapons(WeaponType.AssaultRifle);
        _healthBar.gameObject.SetActive(true);
        transform.position = new Vector3(0, 6, 0);

        if (!IsLocalPlayer)
            return;
        
        UpdatePlayerTeamServerRpc(NetworkObject.NetworkObjectId, team);
        if (team == Lobby.Team.Red)
            transform.position = _world.baseRedPos;
        else if (team == Lobby.Team.Blue)
            transform.position = _world.baseBluePos;
    }
    
    public void DeactivateMouse() {
        mouseActive = false;
        Cursor.visible = true;
        Cursor.lockState = CursorLockMode.None;
    }

    public void ActivateMouse() {
        mouseActive = true;
        Cursor.visible = false;
        Cursor.lockState = CursorLockMode.Locked;
    }
    
    private void OnDisable() {
        GameNetworkManager.UnregisterPlayer(NetworkObject.NetworkObjectId);
    }

    private void Update() {
        if (!IsLocalPlayer || !_world.countdownFinished || !_world.gameStarted.Value)
            return;

        _xAxis = Input.GetAxis("Horizontal");
        _zAxis = Input.GetAxis("Vertical");
        jump = Input.GetButton("Jump");
        _respawn = Input.GetKey(KeyCode.R);

        bool deactivateMouse = _world.gameStarted.Value && Input.GetKeyDown(KeyCode.Escape);
        bool run = mouseActive && Input.GetKey(KeyCode.LeftShift);
        bool destroyBlock = mouseActive && Input.GetMouseButtonDown(0);
        bool buildBlock = mouseActive && Input.GetMouseButtonDown(1);
        bool saveMap = mouseActive && _gameMode == GameMode.Build && Input.GetKeyDown(KeyCode.Z);
        bool loadMap = mouseActive && _gameMode == GameMode.Build && Input.GetKeyDown(KeyCode.U);
        bool assaultRifle = mouseActive && Input.GetKeyDown(KeyCode.Alpha1);
        bool handgun = mouseActive && Input.GetKeyDown(KeyCode.Alpha2);
        bool shovel = mouseActive && Input.GetKeyDown(KeyCode.Alpha3);
        bool iterBlocks = mouseActive && Input.mouseScrollDelta.y < 0;
        bool iterBlocksRev = mouseActive && Input.mouseScrollDelta.y > 0;

        if (mouseActive && deactivateMouse) {
            DeactivateMouse();
        }
        else if (!mouseActive && deactivateMouse) {
            ActivateMouse();
        }

        if (isGrounded)
            currentSpeed = run ? runSpeed : walkSpeed;
        if (destroyBlock) {
            if (_activeWeapon.WeaponType == WeaponType.Shovel)
                PerformRaycastAction(RaycastAction.DestroyBlock, hitRange);
            else
                PerformRaycastAction(RaycastAction.Shoot, _activeWeapon.Range);
        }

        if (buildBlock)
            if (_activeWeapon.WeaponType == WeaponType.Shovel)
                PerformRaycastAction(RaycastAction.BuildBlock, hitRange);

        if (saveMap)
            _saveMapPopup.Open(this);
        if (loadMap)
            _loadMapPopup.Open(this);

        if (handgun)
            SwitchWeapons(WeaponType.Handgun);
        if (assaultRifle)
            SwitchWeapons(WeaponType.AssaultRifle);
        if (shovel)
            SwitchWeapons(WeaponType.Shovel);

        if (iterBlocks) {
            _activeBlock = (BlockType) (((int) _activeBlock + 1) % _inventory.Size);
        }

        if (iterBlocksRev) {
            int nextBlock = ((int) _activeBlock - 1) % _inventory.Size;
            _activeBlock = (BlockType) (nextBlock < 0 ? nextBlock + _inventory.Size : nextBlock); // we have to do this because unity modulo operation is shit
        }

        ProcessMouseInput();
        PerformRaycastAction(RaycastAction.HighlightBlock, hitRange);
    }

    private void LateUpdate() {
        if (!IsLocalPlayer)
            return;
        foreach (GameNetworkManager.PlayerTeam pt in GameNetworkManager.players.Values) {
            pt.player.floatingHealthBar.transform.rotation = Quaternion.LookRotation(pt.player.floatingHealthBar.transform.position - transform.position);
        }
    }
    
    private void FixedUpdate() {
        if (!IsLocalPlayer || !mouseActive)
            return;

        if (_gameMode == GameMode.Build) {
            BuildingModeMovement();
            return;
        }

        CheckAndToggleGrounded();

        // Fall Damage
        if (!_wasFalling && IsFalling) {
            _startOfFall = transform.position.y;
        }

        if (!_wasGrounded && isGrounded) {
            float fallDistance = _startOfFall - transform.position.y;
            if (fallDistance > fallDmgDistThreshold) {
                float fallDamage = (fallDistance - fallDmgDistThreshold) * fallDmgMultiplier;
                TakeDamage(fallDamage);
            }

            //Debug.Log("Fall Distance: " + (_startOfFall - transform.position.y));
        }

        _wasGrounded = isGrounded;
        _wasFalling = IsFalling;

        // Move
        if (isGrounded) {
            _dxz = Vector3.ClampMagnitude(transform.TransformDirection(_xAxis, 0f, _zAxis), 1f);
            _rb.velocity = Vector3.zero; // artificial friction when grounded
        }

        _rb.MovePosition(transform.position + _dxz * (currentSpeed * Time.deltaTime));

        // Jump
        if (jump && isGrounded) {
            _rb.AddForce(jumpForce * _rb.mass * Time.deltaTime * Vector3.up, appliedForceMode);
            isGrounded = false;
        }

        // TODO: remove
        if (_respawn) {
            Respawn();
        }
    }

    private void TakeDamage(float amount) {
        if (!IsLocalPlayer)
            return;
        _audioSource.PlayOneShot(_fallSound);
        _health -= amount;
        _healthBar.value = _health;
        UpdateFloatingHealthBarServerRpc(NetworkObject.NetworkObjectId, _health);
        if (_health < 0) {
            Respawn();
        }
    }

    private void Respawn() {
        Vector3 deathPos = transform.position;
        
        //_rb.velocity = Vector3.zero;
        if (team == Lobby.Team.Red)
            transform.position = _world.baseRedPos;
        else if (team == Lobby.Team.Blue)
            transform.position = _world.baseBluePos;
        _health = maxHealth;
        _healthBar.value = _health;
        UpdateFloatingHealthBarServerRpc(NetworkObject.NetworkObjectId, _health);
        
        _world.PlaceFlag(deathPos);
        DropFlag();
    }
    
    public void PickUpFlag() {
        _hasFlag = true;
        flag.SetActive(true);
        _flagImage.SetActive(true);
    }

    public void DropFlag() {
        _hasFlag = false;
        flag.SetActive(false);
        _flagImage.SetActive(false);
    }
    
    private void SwitchWeapons(WeaponType weapon) {
        _activeWeapon = weapons[weapon];
        PlayerWeaponChangeServerRpc(NetworkObject.NetworkObjectId, weapon);
    }

    private void BuildingModeMovement() {
        bool down = Input.GetKey(KeyCode.LeftShift);
        float yAxis = jump && down || !jump && !down ? 0f : jump ? 1f : -1f;
        _dxz = Vector3.ClampMagnitude(transform.TransformDirection(_xAxis, yAxis, _zAxis), 1f);
        transform.position += _dxz * (currentSpeed * Time.deltaTime);
    }
    
    private void ProcessMouseInput() {
        if (!IsLocalPlayer || !mouseActive) {
            return;
        }

        // Rotate camera around x
        _rotX -= Input.GetAxis("Mouse Y") * SensitivityVer;
        _rotX = Mathf.Clamp(_rotX, MINVert, MAXVert);
        _playerCamera.transform.localEulerAngles = new Vector3(_rotX, 0, 0);

        // Rotate player object around y
        float rotY = Input.GetAxis("Mouse X") * SensitivityHor;
        transform.Rotate(0, rotY, 0);
    }

    private void CheckAndToggleGrounded() {
        if (Physics.Raycast(transform.position, transform.TransformDirection(Vector3.down), out _hit, Mathf.Infinity)) {
            if (string.IsNullOrEmpty(groundTag) || String.Compare(_hit.collider.tag, groundTag, StringComparison.Ordinal) == 0)
                _groundLocation = _hit.point;
            float distanceFromPlayerToGround = Vector3.Distance(transform.position, _groundLocation);
            isGrounded = !(distanceFromPlayerToGround > 0.9f + 0.0001f);
        }
        else {
            isGrounded = false;
            Debug.Log("Error in Player.cs: raycast should always hit an element underneath!");
        }
    }

    private void PlayWeaponSound(Weapon activeWeapon) {
        if (Time.time - _tFired > _activeWeapon.Firerate) {
            if (activeWeapon.WeaponType == WeaponType.Handgun) {
                _audioSync.PlaySound(0);
                //_audioSource.PlayOneShot(_handgunSound);
            }
            else if (activeWeapon.WeaponType == WeaponType.AssaultRifle)
                _audioSync.PlaySound(1);
            //_audioSource.PlayOneShot(_assaultRifleSound);
        }
    }

    private void PerformRaycastAction(RaycastAction raycastAction, float range) {
        Vector3 midPoint = new Vector3(_playerCamera.pixelWidth / 2, _playerCamera.pixelHeight / 2);
        Ray ray = _playerCamera.ScreenPointToRay(midPoint);
        if (Physics.Raycast(ray, out var hit, range)) {
            switch (raycastAction) {
                case RaycastAction.DestroyBlock:
                    GameObject chunk = hit.transform.gameObject;
                    Vector3 localCoordinate = hit.point + (ray.direction / 10000.0f) - chunk.transform.position;
                    chunk.GetComponent<Chunk>().DestroyBlockServerRpc(localCoordinate);
                    byte destBlockId = chunk.GetComponent<Chunk>()
                        .chunkBlocks[Mathf.FloorToInt(localCoordinate.x), Mathf.FloorToInt(localCoordinate.y), Mathf.FloorToInt(localCoordinate.z)].id;
                    _inventory.Add((BlockType) destBlockId);
                    Debug.Log("Inventory at place " + destBlockId % _inventory.Size + " with " + _inventory.Items[destBlockId % _inventory.Size] + " blocks");
                    break;
                case RaycastAction.BuildBlock:
                    if (_inventory.Items[(byte) _activeBlock] > 0) {
                        _inventory.Remove(_activeBlock);
                        _world.BuildBlockServerRpc(hit.point - (ray.direction / 10000.0f), _activeBlock);
                        Debug.Log("Inventory at place " + (byte) _activeBlock % _inventory.Size + " with " +
                                  _inventory.Items[(byte) _activeBlock] + " blocks");
                    }

                    break;
                case RaycastAction.Shoot:
                    if (Time.time - _tFired > _activeWeapon.Firerate) {
                        PlayWeaponSound(_activeWeapon);
                        if (hit.collider.CompareTag(WorldTag)) {
                            chunk = hit.transform.gameObject;
                            localCoordinate = hit.point + (ray.direction / 10000.0f) - chunk.transform.position;
                            Debug.Log("Shoot Block with " + (byte) _activeWeapon.LerpDamage(hit.distance) + " damage");
                            chunk.GetComponent<Chunk>().DamageBlockServerRpc(localCoordinate, (sbyte) _activeWeapon.LerpDamage(hit.distance));
                            //_world.GetComponent<World>().UpdateMeshCollider(chunk);
                        }
                        else if (hit.collider.CompareTag(PlayerTag)) {
                            NetworkObject shotPlayer = hit.collider.gameObject.GetComponent<NetworkObject>();
                            Debug.Log("Shoot Player " + hit.collider.name + " with " + _activeWeapon.LerpDamage(hit.distance) + " damage");
                            if (shotPlayer.GetComponent<Player>().team != team) // Only damage the player if he is not in your team
                                PlayerShotServerRpc(shotPlayer.NetworkObjectId, _activeWeapon.LerpDamage(hit.distance));
                        }

                        _tFired = Time.time;
                    }

                    break;
                case RaycastAction.HighlightBlock:
                    if (_activeWeapon.WeaponType != WeaponType.Shovel || !hit.collider.CompareTag(WorldTag)) {
                        // only highlight blocks when in building mode and when targeting blocks
                        _highlightBlock.SetActive(false);
                        break;
                    }

                    float epsilon = 0.0001f;
                    var blockDim = _highlightBlock.transform.localScale;
                    float blockThickness = blockDim.x;
                    float blockSize = blockDim.y;
                    Vector3 coord = hit.point - (ray.direction / 10000.0f);
                    Vector3 blockPos = new Vector3(Mathf.FloorToInt(coord.x) + blockSize / 2, Mathf.FloorToInt(coord.y) + blockSize / 2,
                        Mathf.FloorToInt(coord.z) + blockSize / 2);
                    if (Math.Abs(hit.point.x - Mathf.Round(hit.point.x)) < epsilon) {
                        // looking at x face
                        _highlightBlock.transform.rotation = Quaternion.Euler(0, 0, 0);
                        blockPos.x += ray.direction.x > 0 ? blockSize / 2 - blockThickness / 2 : -blockSize / 2 + blockThickness / 2;
                    }
                    else if (Math.Abs(hit.point.z - Mathf.Round(hit.point.z)) < epsilon) {
                        // looking at z face
                        _highlightBlock.transform.rotation = Quaternion.Euler(0, 90, 0);
                        blockPos.z += ray.direction.z > 0 ? blockSize / 2 - blockThickness / 2 : -blockSize / 2 + blockThickness / 2;
                    }
                    else {
                        // looking at y face
                        _highlightBlock.transform.rotation = Quaternion.Euler(0, 0, 90);
                        blockPos.y += ray.direction.y > 0 ? blockSize / 2 - blockThickness / 2 : -blockSize / 2 + blockThickness / 2;
                    }

                    _highlightBlock.transform.position = blockPos;
                    _highlightBlock.SetActive(true);
                    break;
                default:
                    Debug.Log("Error in Player.cs: Illegal RaycastAction in method PerformRaycastAction");
                    break;
            }
            // if (hit.transform.gameObject.GetComponent<Chunk>() != null) {}
        }
        else {
            switch (raycastAction) {
                case RaycastAction.HighlightBlock:
                    _highlightBlock.SetActive(false);
                    break;
                case RaycastAction.Shoot:
                    if (Time.time - _tFired > _activeWeapon.Firerate) {
                        PlayWeaponSound(_activeWeapon);
                        _tFired = Time.time;
                    }

                    break;
            }
        }
    }
    
    private void OnGUI() {
        if (!IsLocalPlayer)
            return;
        
        if (_gameMode == GameMode.Fight)
            _inventory.Draw(_activeBlock);
    }
    
    // -- SERVER / CLIENT SYNCHRONIZATION
    
    [ClientRpc]
    private void TakeDamageClientRpc(float amount) {
        Debug.Log("Player " + transform.name + " took " + amount + " damage");
        TakeDamage(amount);
    }

    [ServerRpc]
    private void PlayerShotServerRpc(ulong id, float damage) {
        Player shotPlayer = GameNetworkManager.GetPlayerById(id);
        Debug.Log("Shot Player Game Object " + shotPlayer);
        shotPlayer.TakeDamageClientRpc(damage);
    }
    
    [ServerRpc(RequireOwnership = false)]
    private void PlayerWeaponChangeServerRpc(ulong id, WeaponType weapon) {
        Debug.Log("Server: calling player weapon change for all clients for player with id " + id + " and weapon " + weapon);
        UpdatePlayerWeaponClientRpc(id, weapon);
    }
    
    [ClientRpc]
    private void UpdatePlayerWeaponClientRpc(ulong id, WeaponType weapon) {
        Debug.Log("Updated weapon of player " + id + " on player " + NetworkObject.NetworkObjectId + " to weapon " + weapon.ToString());
        Player target = GameNetworkManager.players[id].player;
        target.weaponModels.ForEach(w => w.SetActive(false));
        foreach (GameObject weaponModel in target.weaponModels) {
            if (weaponModel.transform.name == weapon.ToString())
                weaponModel.SetActive(true);
            if (weaponModel.transform.name == "Cube" && weapon == WeaponType.Shovel) {
                weaponModel.SetActive(true);
            }
        }
    }

    [ServerRpc (RequireOwnership = false)]
    private void UpdateFloatingHealthBarServerRpc(ulong id, float value) {
        Debug.Log("Server call update health bar of player " + id + " to the value " + value);
        UpdateFloatingHealthBarClientRpc(id, value);
    }

    [ClientRpc]
    private void UpdateFloatingHealthBarClientRpc(ulong id, float value) {
        Debug.Log("Client updated health bar of player " + id + " to the value " + value);
        Player target = GameNetworkManager.players[id].player;
        target.floatingHealthBar.value = value;
    }

    [ServerRpc(RequireOwnership = false)]
    private void UpdatePlayerTeamServerRpc(ulong id, Lobby.Team newTeam) {
        UpdatePlayerTeamClientRpc(id, newTeam);
    }

    [ClientRpc]
    private void UpdatePlayerTeamClientRpc(ulong id, Lobby.Team newTeam) {
        Player target = GameNetworkManager.players[id].player;
        target.team = newTeam;
        MeshRenderer meshRenderer = target.GetComponent<MeshRenderer>();
        if (newTeam == Lobby.Team.Blue)
            meshRenderer.material.color = Color.blue;
        else if (newTeam == Lobby.Team.Red)
            meshRenderer.material.color = Color.red;
        else
            Debug.Log("Error in Player.cs: EarlyUpdate(): Player has assigned no team!");
    }
}

public class PlayerInventory : MonoBehaviour {
    public int[] Items => _items;
    public int Size => _items.Length;

    private Dictionary<int, Texture2D> _blockTextures;

    private readonly int[] _items = new int[Enum.GetNames(typeof(BlockType)).Length];
    private readonly GUIStyle _guiStyle = new GUIStyle();
    private readonly GUIStyle _selectedStyle = new GUIStyle();

    private readonly int initX = Screen.width - 100;
    private readonly int initY = Screen.height / 2;
    private readonly int dx = 40;
    private readonly int borderSize = 2;

    public void Start() {
        _blockTextures = new Dictionary<int, Texture2D> {
            {0, Resources.Load<Texture2D>("BlockImages/grass")},
            {1, Resources.Load<Texture2D>("BlockImages/earth")},
            {2, Resources.Load<Texture2D>("BlockImages/iron")},
            {3, Resources.Load<Texture2D>("BlockImages/stone")}
        };
        _guiStyle.fontSize = 30;
        _guiStyle.fontStyle = FontStyle.Bold;
        _guiStyle.normal.textColor = Color.black;

        _selectedStyle.border = new RectOffset(borderSize, borderSize, borderSize, borderSize);
        _selectedStyle.normal.background = Resources.Load<Texture2D>("BlockImages/border");
    }

    public void Add(BlockType blockType) {
        _items[(int) blockType % _items.Length] += 1;
    }

    public void Remove(BlockType blockType) {
        _items[(int) blockType] -= 1;
    }

    public void Draw(BlockType activeBlock) {
        int dy = 0;
        for (int blockType = 0; blockType < _items.Length; blockType++) {
            if (blockType == (int) activeBlock) {
                GUI.Box(new Rect(initX - 5, initY - 5 + dy, _items[blockType].ToString().Length * 20 + 50, 40), GUIContent.none, _selectedStyle);
            }

            GUI.DrawTexture(new Rect(initX, initY + dy, 30, 30), _blockTextures[blockType]);
            GUI.Label(new Rect(initX + dx, initY + dy, 30, 30), _items[blockType].ToString(), _guiStyle);
            dy += 40;
        }
    }
}

public class MapPopup : MonoBehaviour {
    public enum MapPopupAction {
        Load,
        Save
    }

    public MapPopupAction action;

    private World _world;
    private Player _player;
    private Rect windowRect = new Rect((Screen.width - 200) / 2, (Screen.height - 300) / 2, 200, 100);
    private bool show;
    private string mapName; // has to be class variable otherwise it doesn't work!

    private void Start() {
        _world = GameObject.Find("World").GetComponent<World>();
    }

    public void Open(Player player) {
        show = true;
        _player = player;
        _player.popupActive = true;
        _player.DeactivateMouse();
    }

    private void Close() {
        mapName = "";
        show = false;
        _player.popupActive = false;
        _player.ActivateMouse();
    }

    void OnGUI() {
        if (show) {
            windowRect = GUI.Window(0, windowRect, DialogWindow, $"{action.ToString()} Map");
        }
    }

    void DialogWindow(int windowID) {
        GUI.Label(new Rect(5, 20, windowRect.width - 10, 20), "Map Name:");
        mapName = GUI.TextField(new Rect(5, 40, windowRect.width - 10, 20), mapName);

        if (GUI.Button(new Rect(5, 60, windowRect.width - 10, 20), action.ToString())) {
            if (action == MapPopupAction.Load) {
                _world.LoadChunks(mapName);
            }
            else if (action == MapPopupAction.Save) {
                _world.SerializeChunks(mapName);
            }

            Close();
        }

        if (GUI.Button(new Rect(5, 80, windowRect.width - 10, 20), "Cancel")) {
            Close();
        }
    }
}