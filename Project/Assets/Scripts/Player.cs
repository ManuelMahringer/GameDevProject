using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Security;
using TMPro;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Serialization;
using UnityEngine.Networking;
using Unity.Netcode;
using Unity.Netcode.Samples;
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
    private static readonly string PlayerLayerName = "Player";
    private const float HealthBarEnabledAlpha = 180f;
    
    
    [Header("Health & Damage")]
    public float maxHealth = 100f;

    public float fallDmgDistThreshold = 2.5f;
    public float fallDmgMultiplier = 15f;

    [Header("Walk / Run Settings")]
    public float walkSpeed = 3.5f;

    public float runSpeed = 5f;

    [Header("Jump Settings")]
    public float jumpForce = 12_000f;

    public ForceMode appliedForceMode = ForceMode.Force;

    [Header("Build/Destroy Settings")]
    public float hitRangeBuild = 2.5f;

    public float hitRangeDestroy = 2.5f;

    [Header("Ground Tag Specification")]
    public String groundTag = "";

    [Header("Jumping State")]
    [SerializeField]
    private bool jump;

    [SerializeField]
    private bool isGrounded;

    [Header("Current Player Speed")]
    [SerializeField]
    private float currentSpeed;

    public bool popupActive;
    public bool mouseActive;
    public bool active;

    public Lobby.Team team;
    public String playerName;

    private Weapon _activeWeapon;
    private BlockType _activeBlock;
    
    [SerializeField]
    private Color redTeamColor;
    [SerializeField]
    private Color blueTeamColor;
    [SerializeField]
    private float disabledAlpha;

    [SerializeField]
    public List<GameObject> weaponModels;

    [SerializeField]
    private GameObject playerCube;

    [SerializeField]
    public GameObject flag;

    [SerializeField]
    private Material earthMat;

    [SerializeField]
    private Material woodMat;

    [SerializeField]
    private Material stoneMat;

    [SerializeField]
    private Material ironMat;


    // private Animation assaultRifleAnimation = ;

    // private Animation handgunAnimation;

    //  private Animation shovelAnimation;

    //  private Animation placeBlockAnimation;

    private readonly Dictionary<WeaponType, Weapon> weapons = new Dictionary<WeaponType, Weapon>
        {{WeaponType.Handgun, new Handgun()}, {WeaponType.AssaultRifle, new AssaultRifle()}, {WeaponType.Shovel, new Shovel()}};

    private bool IsFalling => !isGrounded && _rb.velocity.y < 0;
    private bool IsWalking => !_dxz.Equals(Vector3.zero) && isGrounded;
    private bool InCountdown => !_world.countdownFinished || !_countdown.countdownFinished;

    private const float SensitivityHor = 5.0f;
    private const float SensitivityVer = 5.0f;
    private const float MINVert = -90.0f;
    private const float MAXVert = 90.0f;

    private AudioSync _audioSync;
    public int spawnOffset;

    // [SerializeField]
    // private Slider healthBar;
    [SerializeField]
    public Slider floatingHealthBar;

    [SerializeField]
    private TMP_Text playerTag; 
    
    private World _world;
    private GameMode _gameMode;
    private Camera _playerCamera;
    private Camera _weaponCamera;
    private float _health;
    private Rigidbody _rb;
    private Slider _healthBar;

    [SerializeField]
    private AudioSource _audioSource;

    [SerializeField]
    private AudioSource _audioSourceWalking;

    private bool _wasInCountdown = false;
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
    private GameObject _hudIngame;
    private GameObject _hudIngameMenu;
    private GameObject _hudGameEnd;
    private Countdown _countdown;
    private TMP_Text _redFlagCntText;
    private TMP_Text _blueFlagCntText;
    private TMP_Text _statusText;
    private TMP_Text _winningMessage;
    private float _statusMsgShow;
    private Button _exitGameBtn;
    private Dictionary<int, Material> _blockMaterials;
    private Image _healthBarFill;

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
        _blockMaterials = new Dictionary<int, Material> {
            {0, earthMat},
            {1, woodMat},
            {2, stoneMat},
            {3, ironMat}
        };

        if (!IsLocalPlayer)
            return;
        floatingHealthBar.maxValue = maxHealth;
        floatingHealthBar.value = maxHealth;
        _audioSync = GetComponent<AudioSync>();
        _gameMode = ComponentManager.gameMode;
        _world = GameObject.Find("World").GetComponent<World>();
        _rb = GetComponent<Rigidbody>();
        _fallSound = Resources.Load("Sounds/hurt_fall") as AudioClip;
        _handgunSound = Resources.Load("Sounds/handgun") as AudioClip;
        _assaultRifleSound = Resources.Load("Sounds/assault_rifle") as AudioClip;
        _healthBar = GameObject.Find("HealthBar").GetComponent<Slider>();
        _health = maxHealth;
        _healthBar.maxValue = _health;
        _healthBar.value = _health;
        _healthBar.interactable = false;
        _healthBar.gameObject.SetActive(false);
        _healthBarFill = _healthBar.GetComponentsInChildren<Image>()[1];
        _highlightBlock = GameObject.Find("Highlight Slab");
        _highlightBlock.SetActive(false);
        _saveMapPopup = gameObject.AddComponent<MapPopup>();
        _saveMapPopup.action = MapPopup.MapPopupAction.Save;
        _loadMapPopup = gameObject.AddComponent<MapPopup>();
        _loadMapPopup.action = MapPopup.MapPopupAction.Load;
        _activeBlock = BlockType.Earth;
        _inventory = gameObject.AddComponent<PlayerInventory>();
        _flagImage = GameObject.Find("FlagImage");
        _flagImage.SetActive(false);
        _hudIngame = GameObject.Find("HUD (ingame)");
        _hudGameEnd = GameObject.Find("HUD (game end)");
        _hudIngameMenu = GameObject.Find("HUD (ingame menu)");
        _countdown = GameObject.Find("HUD (Countdown)").GetComponent<Countdown>();
        _redFlagCntText = GameObject.Find("FlagsRedText").GetComponent<TMP_Text>();
        _blueFlagCntText = GameObject.Find("FlagsBlueText").GetComponent<TMP_Text>();
        _statusText = GameObject.Find("StatusText").GetComponent<TMP_Text>();
        _winningMessage = GameObject.Find("WinningMessage").GetComponent<TMP_Text>();
        _exitGameBtn = GameObject.Find("ExitGameButton").GetComponent<Button>();
        
        _inventory.Initialize();
        _inventory.Active(false);
        _hudIngame.SetActive(false);
        _hudGameEnd.SetActive(false);
        floatingHealthBar.gameObject.SetActive(false);
        InitWeaponModels();

        Camera[] cameras = gameObject.GetComponentsInChildren<Camera>();
        _playerCamera = cameras[0];
        _playerCamera.enabled = true;
        _weaponCamera = cameras[1];
        _weaponCamera.enabled = true;

        playerCube.SetActive(false);

        if (_gameMode == GameMode.Build) {
            _rb.isKinematic = true;
            hitRangeBuild = Single.PositiveInfinity;
            isGrounded = true;
            _healthBar.gameObject.SetActive(false);
            runSpeed = walkSpeed = 8f;
            for (int i = 0; i < _inventory.Size; i++) {
                _inventory.Items[i] = Int32.MaxValue / 2;
            }
        }

        DeactivateMouse();
        
        _world.gameStarted.OnValueChanged += OnGameStarted;
        _world.redFlagCnt.OnValueChanged += OnRedFlagCntChanged;
        _world.blueFlagCnt.OnValueChanged += OnBlueFlagCntChanged;
        _world.gameEnded.OnValueChanged += OnGameEnded;
        _world.flagHolderId.OnValueChanged += OnNewFlagHolder;
        _world.respawnDirtyFlagState.OnValueChanged = OnDirtyFlagStateSet;
        _world.hostQuit.OnValueChanged = OnHostQuit;
    }

    private void OnGameStarted(bool oldVal, bool newVal) {
        ActivateMouse();
        SwitchWeapons(WeaponType.AssaultRifle);
        _healthBar.gameObject.SetActive(true);
        gameObject.GetComponentInChildren<AudioListener>().enabled = true;

        if (!IsLocalPlayer)
            return;

        _hudIngame.SetActive(true);
        _healthBarFill.color = team == Lobby.Team.Blue ? blueTeamColor : redTeamColor;
        UpdatePlayerTeamServerRpc(NetworkObjectId, team);
        UpdatePlayerTagServerRpc(NetworkObjectId, playerName);
        _inventory.Active(true);
        if (team == Lobby.Team.Red) {
            transform.position = _world.baseRedPos + new Vector3(0,0, spawnOffset*5);
            transform.rotation = Quaternion.Euler(0, 90, 0);
        }
        else if (team == Lobby.Team.Blue) {
            Debug.Log("spawning at baseBluePos " +_world.baseBluePos.z + " and offset " + spawnOffset*5);
            transform.position = _world.baseBluePos + new Vector3(0,0, spawnOffset*5);;
            transform.rotation = Quaternion.Euler(0, -90, 0);
        }

        _rb.constraints = RigidbodyConstraints.FreezeAll;
        _rb.velocity = Vector3.zero;
    }

    private void OnGameEnded(bool oldVal, bool newVal) {
        _hudIngame.SetActive(false);
        //_inventory.active = false;
        _inventory.Active(false);
        Lobby.Team winning = _world.redFlagCnt.Value == _world.capturesToWin ? Lobby.Team.Red : Lobby.Team.Blue;
        _winningMessage.color = winning == Lobby.Team.Red ? Color.red : Color.blue;
        _winningMessage.text = winning + " won!";
        _hudGameEnd.SetActive(true);
        DeactivateMouse();
    }

    private void OnHostQuit(bool oldVal, bool newVal) {
        if (!IsLocalPlayer)
            return;
        if (newVal) {
            Application.Quit();
        }
    }

    private void OnApplicationQuit() {
        if (IsHost) {
            Debug.Log("Setting application quit");
            _world.hostQuit.Value = true;
        }
    }

    private void OnNewFlagHolder(ulong oldId, ulong newId) {
        if (IsLocalPlayer && NetworkObject.NetworkObjectId == newId) {
            Debug.Log("Local Player " + NetworkObject.NetworkObjectId + ": UI flag activated");
            _flagImage.SetActive(true);
        }
        else {
            Debug.Log("Local Player " + NetworkObject.NetworkObjectId + ": UI flag deactivated");
            _flagImage.SetActive(false);
        }
    }

    private void OnRedFlagCntChanged(int oldVal, int newVal) {
        _redFlagCntText.text = newVal.ToString();
    }

    private void OnBlueFlagCntChanged(int oldVal, int newVal) {
        _blueFlagCntText.text = newVal.ToString();
    }

    public void DeactivateMouse() {
        mouseActive = false;
        Cursor.visible = true;
        Cursor.lockState = CursorLockMode.None;
        Debug.Log("Stopping sound");
        _audioSync.StopSoundLoop();
    }

    public void ActivateMouse() {
        mouseActive = true;
        Cursor.visible = false;
        Cursor.lockState = CursorLockMode.Locked;
    }

    public void EnableIngameHud(bool enable) {
        _hudIngame.SetActive(enable);
        _inventory.Active(enable);
    }

    private void InitWeaponModels() {
        foreach (var model in weaponModels) {
            model.layer = LayerMask.NameToLayer(WeaponLayerName);
            if (model.transform.name == WeaponType.Shovel.ToString()) {
                model.transform.GetChild(0).gameObject.layer = LayerMask.NameToLayer(WeaponLayerName);
                model.transform.GetChild(1).gameObject.layer = LayerMask.NameToLayer(WeaponLayerName);
            }

            if (model.transform.name == WeaponType.AssaultRifle.ToString()) {
                model.transform.GetChild(0).gameObject.layer = LayerMask.NameToLayer(WeaponLayerName);
                model.transform.GetChild(1).gameObject.layer = LayerMask.NameToLayer(WeaponLayerName);
                model.transform.GetChild(2).gameObject.layer = LayerMask.NameToLayer(WeaponLayerName);
                model.transform.GetChild(3).gameObject.layer = LayerMask.NameToLayer(WeaponLayerName);
            }

            if (model.transform.name == WeaponType.Handgun.ToString()) {
                model.transform.GetChild(0).gameObject.layer = LayerMask.NameToLayer(WeaponLayerName);
                model.transform.GetChild(1).gameObject.layer = LayerMask.NameToLayer(WeaponLayerName);
                model.transform.GetChild(2).gameObject.layer = LayerMask.NameToLayer(WeaponLayerName);
                model.transform.GetChild(3).gameObject.layer = LayerMask.NameToLayer(WeaponLayerName);
            }
        }
    }

    private void OnDisable() {
        GameNetworkManager.UnregisterPlayer(NetworkObject.NetworkObjectId);
    }

    private void Update() {
        if (!IsLocalPlayer)
            return;

        _xAxis = Input.GetAxis("Horizontal");
        _zAxis = Input.GetAxis("Vertical");
        jump = Input.GetButton("Jump");
        //_respawn = Input.GetKey(KeyCode.R);

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

        if (Input.GetKeyDown(KeyCode.L))
            Debug.Log("This player network object id: " + NetworkObjectId);

        if (mouseActive && deactivateMouse) {
            _hudIngameMenu.GetComponent<InGameMenu>().Show(true, this);
            DeactivateMouse();
        }
        else if (!mouseActive && deactivateMouse) {
            _hudIngameMenu.GetComponent<InGameMenu>().Show(false, this);
            ActivateMouse();
        }

        if (!InCountdown && _wasInCountdown) {
            _healthBarFill.color = team == Lobby.Team.Blue ? blueTeamColor : redTeamColor; 
            UpdateFloatingHealthBarServerRpc(NetworkObjectId, _health);
            UpdatePlayerTagServerRpc(NetworkObjectId, playerName);
            _rb.constraints = RigidbodyConstraints.FreezeRotation;
        }
        _wasInCountdown = InCountdown;
        
        if (InCountdown)
            return;
        
        if (isGrounded)
            currentSpeed = run ? runSpeed : walkSpeed;
        if (destroyBlock) {
            if (_activeWeapon.WeaponType == WeaponType.Shovel) {
                PerformRaycastAction(RaycastAction.DestroyBlock, _activeWeapon.Range);
            }
            else {
                PerformRaycastAction(RaycastAction.Shoot, _activeWeapon.Range);
            }
        }

        if (buildBlock)
            if (_activeWeapon.WeaponType == WeaponType.Shovel)
                PerformRaycastAction(RaycastAction.BuildBlock, _activeWeapon.Range);

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
            if (_activeWeapon.WeaponType == WeaponType.Shovel)
                UpdatePlayerCubeServerRpc(NetworkObjectId, _inventory.Items[(int) _activeBlock] > 0, _activeBlock);
        }

        if (iterBlocksRev) {
            int nextBlock = ((int) _activeBlock - 1) % _inventory.Size;
            _activeBlock = (BlockType) (nextBlock < 0 ? nextBlock + _inventory.Size : nextBlock); // we have to do this because unity modulo operation is shit
            if (_activeWeapon.WeaponType == WeaponType.Shovel)
                UpdatePlayerCubeServerRpc(NetworkObjectId, _inventory.Items[(int) _activeBlock] > 0, _activeBlock);
        }

        if (IsWalking && !_audioSourceWalking.isPlaying) {
            _audioSync.StartSoundLoop();
        }
        else if (!IsWalking) {
            _audioSync.StopSoundLoop();
        }
        
        // Reset status message after show time
        if (Time.time - _statusMsgShow > _world.statusMsgShowTime)
            _statusText.text = "";
        
        ProcessMouseInput();
        PerformRaycastAction(RaycastAction.HighlightBlock, hitRangeBuild);
    }
    

    private void LateUpdate() {
        if (!IsLocalPlayer)
            return;
        foreach (GameNetworkManager.PlayerTeam pt in GameNetworkManager.players.Values) {
            pt.player.floatingHealthBar.transform.rotation = Quaternion.LookRotation(pt.player.floatingHealthBar.transform.position - transform.position);
        }
    }

    private void FixedUpdate() {
        if (!IsLocalPlayer ) //|| !mouseActive || InCountdown
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
                _audioSync.PlaySound(3);
            }
            else {
                _audioSync.PlaySound(2);
            }

            //Debug.Log("Fall Distance: " + (_startOfFall - transform.position.y));
        }

        _wasGrounded = isGrounded;
        _wasFalling = IsFalling;

        // Move
        _dxz = Vector3.ClampMagnitude(transform.TransformDirection(_xAxis, 0f, _zAxis), 1f);
        if (isGrounded) {
            // _dxz = Vector3.ClampMagnitude(transform.TransformDirection(_xAxis, 0f, _zAxis), 1f);
            _rb.velocity = Vector3.zero; // artificial friction when grounded
        }

        if (!mouseActive || InCountdown) {
            _dxz = Vector3.zero;
        }

        _rb.MovePosition(transform.position + _dxz * (currentSpeed * Time.deltaTime));

        // Jump
        if (jump && isGrounded) {
            _rb.AddForce(jumpForce * _rb.mass * Time.deltaTime * Vector3.up, appliedForceMode);
            isGrounded = false;
        }
    }

    private void TakeDamage(float amount) {
        if (!IsLocalPlayer || InCountdown)
            return;
        //_audioSource.PlayOneShot(_fallSound);
        _health -= amount;
        _healthBar.value = _health;
        UpdateFloatingHealthBarServerRpc(NetworkObject.NetworkObjectId, _health);
        if (_health <= 0) {
            Respawn();
        }
    }

    private void Respawn() {
        if (!IsLocalPlayer)
            return;

        if (_world.flagHolderId.Value == NetworkObjectId) {
            _respawn = true;
            _world.SetDirtyFlagStateServerRpc();
            return;
        }

        ResetOnSpawn();
    }

    private void OnDirtyFlagStateSet(bool oldVal, bool newVal) {
        if (!newVal || !_respawn)
            return;
        _respawn = false;
        _world.DropFlagServerRpc(NetworkObjectId, transform.position);
        
        ResetOnSpawn();
        
        _world.PlayerResetCallbackServerRpc(NetworkObjectId);
    }
    
    private void ResetOnSpawn() {
        // Reset health bar
        _health = maxHealth;
        _healthBar.value = _health;
        _healthBarFill.color = new Color(_healthBarFill.color.r, _healthBarFill.color.b, _healthBarFill.color.g, disabledAlpha/255);
        UpdateFloatingHealthBarServerRpc(NetworkObjectId, _health, disabledAlpha);
        UpdatePlayerTagServerRpc(NetworkObjectId, playerName, disabledAlpha);
        
        // Initiate respawn countdown
        _world.countdownFinished = false;
        _countdown.GetComponent<Countdown>().StartLocalCountdown("Respawning in ...");
        
        // Reset player position and lookAt
        _rotX = 0;
        _playerCamera.transform.localEulerAngles = new Vector3(_rotX, 0, 0);
        if (team == Lobby.Team.Red) {
            transform.position = _world.baseRedPos + new Vector3(0,0, spawnOffset * 5);
            transform.rotation = Quaternion.Euler(0, 90, 0);
            //transform.position = new Vector3(-5, 4.5f, 0);
        }
        else if (team == Lobby.Team.Blue) {
            transform.position = _world.baseBluePos + new Vector3(0,0, spawnOffset * 5);;
            transform.rotation = Quaternion.Euler(0, -90, 0);
            //transform.position = new Vector3(5, 4.5f, 0);
        }

        _rb.constraints = RigidbodyConstraints.FreezeAll;
        _rb.velocity = Vector3.zero;
        
        // Reset inventory
        _inventory.Clear();
        UpdatePlayerCubeServerRpc(NetworkObjectId, false, _activeBlock);
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
            isGrounded = distanceFromPlayerToGround <= 0.9f + 0.00001f;
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
            else if (activeWeapon.WeaponType == WeaponType.Shovel)
               _audioSync.PlaySound(4); 
            //_audioSource.PlayOneShot(_shovelSound);
        }
    }

    private void PlayAnimation(Weapon activeWeapon, bool placeBlock = false, bool melee = false) {
        foreach (GameObject gameObject in weaponModels) {

            Animation anim = gameObject.GetComponent<Animation>();
            String name = gameObject.transform.name;

            // block is placed so just play this animation
            if (placeBlock) {
                if (name == "Cube") {
                    anim.Stop();
                    anim.Play("PlaceBlock");
                }
            }
            // literally every other animation
            else {
                // melee hit 
                if (name == "Shovel" && melee){
                    // find attack script and play
                    anim.Stop();
                    anim.Play("AttackShovel");
                }
                else if (name == activeWeapon.WeaponType.ToString()) {
                    anim.Stop();
                    anim.Play();    // plays default Animation so no need to specify
                }
            }
        }
    }

    private bool CheckProtectedZone(Ray ray, Vector3 hit, RaycastAction action) {
        Vector3 center = hit;
        switch (action) {
            case RaycastAction.DestroyBlock:
                center += (ray.direction / 10000.0f);
                break;
            case RaycastAction.BuildBlock:
                center -= (ray.direction / 10000.0f);
                break;
            default:
                Debug.Log("Error in Player.cs: Illegal RaycastAction in method PerformRaycastAction");
                break;
        }

        center = new Vector3(Mathf.FloorToInt(center.x) + 0.5f, Mathf.FloorToInt(center.y), Mathf.FloorToInt(center.z) + 0.5f);
        bool inProtectedZone = _world.InProtectedZone(center);
        if (inProtectedZone) {
            _statusMsgShow = Time.time;
            _statusText.text = "Can't build / destroy in protected zone";
        }
        return inProtectedZone;
    }

    private void PerformRaycastAction(RaycastAction raycastAction, float range) {
        Vector3 midPoint = new Vector3(_playerCamera.pixelWidth / 2, _playerCamera.pixelHeight / 2);
        Ray ray = _playerCamera.ScreenPointToRay(midPoint);
        // Ignore protection layers when raycasting
        if (Physics.Raycast(ray, out var hit, range, ~(1 << LayerMask.NameToLayer(_world.protectionLayerName)))) {
            switch (raycastAction) {
                case RaycastAction.DestroyBlock:
                    // Melee hit
                    if (hit.collider.CompareTag(PlayerTag)) {
                        PerformRaycastAction(RaycastAction.Shoot, _activeWeapon.Range);
                        PlayAnimation(_activeWeapon, false, true);
                        PlayWeaponSound(_activeWeapon);
                        break;
                    }
                    GameObject chunk = hit.transform.gameObject;

                    // Can't destroy a block in a protected radius
                    if (CheckProtectedZone(ray, hit.point, raycastAction))
                        break;
                    
                    Vector3 localCoordinate = hit.point + (ray.direction / 10000.0f) - chunk.transform.position;
                    chunk.GetComponent<Chunk>().DestroyBlockServerRpc(localCoordinate);
                    byte destBlockId = chunk.GetComponent<Chunk>()
                        .chunkBlocks[Mathf.FloorToInt(localCoordinate.x), Mathf.FloorToInt(localCoordinate.y), Mathf.FloorToInt(localCoordinate.z)].id;
                    _inventory.Add((BlockType) destBlockId);
                    PlayAnimation(_activeWeapon);
                    Debug.Log("Inventory at place " + destBlockId % _inventory.Size + " with " + _inventory.Items[destBlockId % _inventory.Size] + " blocks");
                    if (destBlockId == (int) _activeBlock)
                        UpdatePlayerCubeServerRpc(NetworkObjectId, true, _activeBlock);
                    break;
                case RaycastAction.BuildBlock:
                    // Check if block is at max world height
                    if (hit.point.y >= _world.height * _world.chunkSize)
                        break;

                    if (_inventory.Items[(byte) _activeBlock] > 0) {
                        // Can't build inside protection zone
                        if (CheckProtectedZone(ray, hit.point, raycastAction))
                            break;
                        // Can't build where player is standing
                        Vector3 buildPos = hit.point - (ray.direction / 10000.0f);
                        if (Physics.CheckBox(buildPos, new Vector3(0.45f, 0.45f, 0.45f), Quaternion.identity, 1 << LayerMask.NameToLayer(PlayerLayerName)))
                            break;
                        
                        _inventory.Remove(_activeBlock);
                        _world.BuildBlockServerRpc(hit.point - (ray.direction / 10000.0f), _activeBlock);
                        PlayAnimation(_activeWeapon, true);
                        Debug.Log("Inventory at place " + (byte) _activeBlock % _inventory.Size + " with " +
                                  _inventory.Items[(byte) _activeBlock] + " blocks");
                        if (_inventory.Items[(int) _activeBlock] == 0) {
                            UpdatePlayerCubeServerRpc(NetworkObjectId, false, _activeBlock);
                        }
                    }

                    break;
                case RaycastAction.Shoot:
                    if (Time.time - _tFired > _activeWeapon.Firerate) {
                        PlayWeaponSound(_activeWeapon);
                        PlayAnimation(_activeWeapon);
                        if (hit.collider.CompareTag(WorldTag)) {
                            // Can't destroy a block in a protected radius
                            if (CheckProtectedZone(ray, hit.point, raycastAction))
                                break;
                            
                            chunk = hit.transform.gameObject;
                            localCoordinate = hit.point + (ray.direction / 10000.0f) - chunk.transform.position;
                            Debug.Log("Shoot Block with " + (byte) _activeWeapon.LerpDamage(hit.distance) + " damage");
                            chunk.GetComponent<Chunk>().DamageBlockServerRpc(localCoordinate, (sbyte) _activeWeapon.LerpDamage(hit.distance));
                            //_world.GetComponent<World>().UpdateMeshCollider(chunk);
                        }
                        else if (hit.collider.CompareTag(PlayerTag)) {
                            NetworkObject shotPlayer = hit.collider.gameObject.GetComponent<NetworkObject>();
                            float damage = _activeWeapon.WeaponType == WeaponType.Shovel ? _activeWeapon.Damage : _activeWeapon.LerpDamage(hit.distance);
                            Debug.Log("Shoot Player " + hit.collider.name + " with " + damage + " damage");
                            if (shotPlayer.GetComponent<Player>().team != team) // Only damage the player if he is not in your team
                                PlayerShotServerRpc(shotPlayer.NetworkObjectId, damage);
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
                        PlayAnimation(_activeWeapon);
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
        // Debug.Log("Shot Player Game Object " + shotPlayer);
        shotPlayer.TakeDamageClientRpc(damage);
    }

    [ServerRpc(RequireOwnership = false)]
    private void PlayerWeaponChangeServerRpc(ulong id, WeaponType weapon) {
        // Debug.Log("Server: calling player weapon change for all clients for player with id " + id + " and weapon " + weapon);
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
                UpdatePlayerCubeServerRpc(id, target._inventory.Items[(int) _activeBlock] > 0, target._activeBlock);
            }
        }
    }

    [ServerRpc(RequireOwnership = false)]
    private void UpdatePlayerCubeServerRpc(ulong id, bool active, BlockType type) {
        UpdatePlayerCubeClientRpc(id, active, type);
    }

    [ClientRpc]
    private void UpdatePlayerCubeClientRpc(ulong id, bool active, BlockType type) {
        Player target = GameNetworkManager.players[id].player;
        if (active) {
            // Reset the block position if it may be stuck in animation
            target.playerCube.transform.localPosition = Vector3.zero;
            target.playerCube.transform.localEulerAngles = Vector3.zero;
            target.playerCube.SetActive(true);
            //Debug.Log("Setting cube material of player " + target + " to block " + type);
            target.playerCube.GetComponent<MeshRenderer>().material = _blockMaterials[(int) type];
        }
        else {
            target.playerCube.SetActive(false);
        }
    }

    [ServerRpc(RequireOwnership = false)]
    private void UpdateFloatingHealthBarServerRpc(ulong id, float value, float alpha = HealthBarEnabledAlpha) {
        //Debug.Log("Server call update health bar of player " + id + " to the value " + value);
        UpdateFloatingHealthBarClientRpc(id, value, alpha);
    }

    [ClientRpc]
    private void UpdateFloatingHealthBarClientRpc(ulong id, float value, float alpha) {
        //Debug.Log("Client updated health bar of player " + id + " to the value " + value);
        Player target = GameNetworkManager.players[id].player;
        target.floatingHealthBar.value = value;
        Image fill = target.floatingHealthBar.GetComponentsInChildren<Image>()[1];
        fill.color = new Color(fill.color.r, fill.color.g, fill.color.b, alpha/255);
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
        GameObject handgun = target.weaponModels.Find(m => m.name == WeaponType.Handgun.ToString());
        GameObject assaultRifle = target.weaponModels.Find(m => m.name == WeaponType.AssaultRifle.ToString());
        GameObject shovel = target.weaponModels.Find(m => m.name == WeaponType.Shovel.ToString());
        MeshRenderer handgunSled = handgun.GetComponentsInChildren<MeshRenderer>()[3];
        MeshRenderer assaultRifleDustcover = assaultRifle.GetComponentsInChildren<MeshRenderer>()[1];
        MeshRenderer shovelHandle = shovel.GetComponentsInChildren<MeshRenderer>()[0];
        Image floatingHealthBarFill = target.floatingHealthBar.GetComponentsInChildren<Image>()[1];
        if (newTeam == Lobby.Team.Blue) {
            meshRenderer.material.color = blueTeamColor;
            floatingHealthBarFill.color = blueTeamColor;
            handgunSled.material.color = blueTeamColor;
            assaultRifleDustcover.material.color = blueTeamColor;
            shovelHandle.material.color = blueTeamColor;
        }
        else if (newTeam == Lobby.Team.Red) {
            meshRenderer.material.color = redTeamColor;
            floatingHealthBarFill.color = redTeamColor;
            handgunSled.material.color = redTeamColor;
            assaultRifleDustcover.material.color = redTeamColor;
            shovelHandle.material.color = redTeamColor;
        }
        else
            Debug.Log("Error in Player.cs: EarlyUpdate(): Player has assigned no team!");
    }

    [ServerRpc(RequireOwnership = false)]
    private void UpdatePlayerTagServerRpc(ulong id, string t, float alpha = 255f) {
        UpdatePlayerTagClientRpc(id, t, alpha);
    }

    [ClientRpc]
    private void UpdatePlayerTagClientRpc(ulong id, string t, float alpha) {
        Player target = GameNetworkManager.GetPlayerById(id);
        target.playerTag.text = t;
        target.playerTag.color = new Color(target.playerTag.color.r, target.playerTag.color.b, target.playerTag.color.g, alpha/255);
    }
}

public class PlayerInventory : MonoBehaviour {
    public int[] Items => _items;
    public int Size => _items.Length;

    private Dictionary<int, Image> _blockImages;
    private Dictionary<int, TMP_Text> _blockCounts;

    private readonly int[] _items = new int[Enum.GetNames(typeof(BlockType)).Length];
    private readonly GUIStyle _selectedStyle = new GUIStyle();

    private bool _active;

    private readonly int borderSize = 2;

    public void Start() {
        _selectedStyle.border = new RectOffset(borderSize, borderSize, borderSize, borderSize);
        _selectedStyle.normal.background = Resources.Load<Texture2D>("BlockImages/border");
    }

    public void Active(bool active) {
        _active = active;
        foreach (var blockImage in _blockImages.Values) {
            blockImage.gameObject.SetActive(active);
        }
        foreach (var blockCount in _blockCounts.Values) {
            blockCount.gameObject.SetActive(active);
        }
    }

    public void Initialize() {
        _blockImages = new Dictionary<int, Image> {
            {0, GameObject.Find("BlockEarth").GetComponent<Image>()},
            {1, GameObject.Find("BlockWood").GetComponent<Image>()},
            {2, GameObject.Find("BlockStone").GetComponent<Image>()},
            {3, GameObject.Find("BlockIron").GetComponent<Image>()}
        };

        _blockCounts = new Dictionary<int, TMP_Text> {
            {0, GameObject.Find("BlockEarthCount").GetComponent<TMP_Text>()},
            {1, GameObject.Find("BlockWoodCount").GetComponent<TMP_Text>()},
            {2, GameObject.Find("BlockStoneCount").GetComponent<TMP_Text>()},
            {3, GameObject.Find("BlockIronCount").GetComponent<TMP_Text>()}
        };
    }

    public void Add(BlockType blockType) {
        _items[(int) blockType % _items.Length] += 1;
        _blockCounts[(int) blockType].text = _items[(int) blockType].ToString();
    }

    public void Remove(BlockType blockType) {
        _items[(int) blockType] -= 1;
        _blockCounts[(int) blockType].text = _items[(int) blockType].ToString();
    }

    public void Clear() {
        for (int i = 0; i < _items.Length; i++)
            _items[i] = 0;
    }

    public void Draw(BlockType activeBlock) {
        if (!_active) {
            Debug.Log("Inventory not active");
            return;
        }

        for (int blockType = 0; blockType < _items.Length; blockType++) {
            if (blockType == (int) activeBlock) {
                Image active = _blockImages[blockType];
                Vector3 imgPos = active.transform.position;
                GUI.Box(new Rect(imgPos.x - 23, Screen.height - imgPos.y - 23, 110, 23 * 2), GUIContent.none, _selectedStyle);
            }
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