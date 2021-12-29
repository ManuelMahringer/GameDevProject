using System;
using System.Collections;
using System.Collections.Generic;
using System.Security;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Serialization;
using UnityEngine.Networking;
using Unity.Netcode;
using UnityEditor;

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
    public float hitRange = 5f;

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

    private bool IsFalling => !isGrounded && _rb.velocity.y < 0;

    private const float SensitivityHor = 5.0f;
    private const float SensitivityVer = 5.0f;
    private const float MINVert = -90.0f;
    private const float MAXVert = 90.0f;

    private World _world;
    private GameMode _gameMode;
    private Camera _playerCamera;
    private float _health;
    private Rigidbody _rb;
    private Camera _camera;
    private Slider _healthBar;
    private AudioSource _audioSource;
    private AudioClip _fallSound;
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
    
    private enum RaycastAction {
        DestroyBlock,
        BuildBlock,
        Shoot,
        HighlightBlock
    }

    public void TakeDamage(float amount) {
        Debug.Log("Take Damage: " + amount);
        _audioSource.PlayOneShot(_fallSound);
        _health -= amount;
        _healthBar.value = _health;
        if (_health < 0) {
            Die();
        }
    }

    private void Die() {
        Debug.Log("PLAYER DIES");
    }

    private void Start() {
        if (!IsLocalPlayer)
            return;
        _gameMode = ComponentManager.gameMode;
        _world = GameObject.Find("World").GetComponent<World>();
        _rb = GetComponent<Rigidbody>();
        _audioSource = GetComponent<AudioSource>();
        _fallSound = Resources.Load("Sounds/hurt_fall") as AudioClip;
        _healthBar = GameObject.Find("HealthBar").GetComponent<Slider>();
        _health = maxHealth;
        _healthBar.maxValue = _health;
        _healthBar.value = _health;
        _highlightBlock = GameObject.Find("Highlight Slab");
        _saveMapPopup = gameObject.AddComponent<MapPopup>();
        _saveMapPopup.action = MapPopup.MapPopupAction.Save;
        _loadMapPopup = gameObject.AddComponent<MapPopup>();
        _loadMapPopup.action = MapPopup.MapPopupAction.Load;
        
        _playerCamera = GetComponentInChildren<Camera>();
        mouseActive = true;
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
        
        transform.position = new Vector3(0, 7, 0);

        Debug.Log("before is owner" + IsOwner + IsLocalPlayer);
        if (!IsOwner)
            return;
        Debug.Log(" after  is owner");
        _camera = gameObject.GetComponentInChildren<Camera>();
        _camera.enabled = true;
        Debug.Log("Camera enabled");
        gameObject.GetComponentInChildren<AudioListener>().enabled = true;

        if (_gameMode == GameMode.Build) {
            _rb.isKinematic = true;
            hitRange = Single.PositiveInfinity;
            isGrounded = true;
            _healthBar.gameObject.SetActive(false);
            runSpeed = walkSpeed = 8f;
        }
    }


    private void Update() {
        //Debug.Log("lcoalplayer?" + IsLocalPlayer);
        if (!IsLocalPlayer)
            return;
        //Debug.Log("Afterlocalplayer");
        _xAxis = Input.GetAxis("Horizontal");
        _zAxis = Input.GetAxis("Vertical");
        jump = Input.GetButton("Jump");
        _respawn = Input.GetKey(KeyCode.R);

        bool run = Input.GetKey(KeyCode.LeftShift);
        bool destroyBlock = !popupActive && Input.GetMouseButtonDown(0);
        bool buildBlock = !popupActive && Input.GetMouseButtonDown(1);
        bool shoot = Input.GetKeyDown(KeyCode.T);
        bool rpc_call = Input.GetKeyDown(KeyCode.U);
        bool saveMap = Input.GetKeyDown(KeyCode.Z);
        bool loadMap = Input.GetKeyDown(KeyCode.U);
        bool deactivateMouse = Input.GetKeyDown(KeyCode.Escape);
        
        if (isGrounded)
            currentSpeed = run ? runSpeed : walkSpeed;
        if (destroyBlock)
            PerformRaycastAction(RaycastAction.DestroyBlock, hitRange);
        if (buildBlock)
            PerformRaycastAction(RaycastAction.BuildBlock, hitRange);
        if (shoot)
            PerformRaycastAction(RaycastAction.Shoot, float.PositiveInfinity);
        if (rpc_call) {
            Debug.Log("RPCCalled");
            //GameObject chunk = GameObject.Find("World").GetComponent<World>().FindChunkServerRpc(new Vector3(0,1,0));
            //chunk.GetComponent<Chunk>().PingServerRpc("TestRPC");
        }
        if (saveMap)
            _saveMapPopup.Open(this);
        if (loadMap)
            _loadMapPopup.Open(this);
        if (mouseActive && deactivateMouse) {
            DeactivateMouse();
        } else if (!mouseActive && deactivateMouse) {
            ActivateMouse();
        }

        if (Input.GetKeyDown(KeyCode.J)) {
            Debug.Log(_world.testNetworkVar.Value);
        }

        if (Input.GetKeyDown(KeyCode.K)) {
            _world.SetTestNetworkVarServerRpc('K');
        }

        if (Input.GetKeyDown(KeyCode.L)) {
            _world.SetTestNetworkVarServerRpc('L');
        }

        ProcessMouseInput();
        PerformRaycastAction(RaycastAction.HighlightBlock, hitRange);
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

    private void ProcessMouseInput() {
        if (!IsOwner || !mouseActive) {
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

    private void FixedUpdate() {
        if (!IsLocalPlayer)
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

            Debug.Log("Fall Distance: " + (_startOfFall - transform.position.y));
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
            _rb.velocity = Vector3.zero;
            _rb.position = new Vector3(0, 7, 0);
            _health = maxHealth;
            _healthBar.value = _health;
        }
    }

    private void BuildingModeMovement() {
        bool down = Input.GetKey(KeyCode.LeftShift);
        float yAxis = jump && down || !jump && !down ? 0f : jump ? 1f : -1f;
        _dxz = Vector3.ClampMagnitude(transform.TransformDirection(_xAxis, yAxis, _zAxis), 1f);
        transform.position += _dxz * (currentSpeed * Time.deltaTime);
    }

    private void CheckAndToggleGrounded() {
        if (Physics.Raycast(transform.position, transform.TransformDirection(Vector3.down), out _hit, Mathf.Infinity)) {
            if (string.IsNullOrEmpty(groundTag) || String.Compare(_hit.collider.tag, groundTag, StringComparison.Ordinal) == 0)
                _groundLocation = _hit.point;
            float distanceFromPlayerToGround = Vector3.Distance(transform.position, _groundLocation);
            isGrounded = !(distanceFromPlayerToGround > 1f + 0.0001f);
        }
        else {
            isGrounded = false;
            Debug.Log("Error in Player.cs: raycast should always hit an element underneath!");
        }
    }

    private void PerformRaycastAction(RaycastAction raycastAction, float range) {
        Vector3 midPoint = new Vector3(_camera.pixelWidth / 2, _camera.pixelHeight / 2);
        Ray ray = _camera.ScreenPointToRay(midPoint);
        if (Physics.Raycast(ray, out var hit, range)) {
            switch (raycastAction) {
                case RaycastAction.DestroyBlock:
                    GameObject chunk = hit.transform.gameObject;
                    Vector3 localCoordinate = hit.point + (ray.direction / 10000.0f) - chunk.transform.position;
                    //chunk.GetComponent<Chunk>().DestroyBlock(localCoordinate);
                    chunk.GetComponent<Chunk>().DestroyBlockServerRpc(localCoordinate);
                    //_world.GetComponent<World>().UpdateMeshCollider(chunk);
                    break;
                case RaycastAction.BuildBlock:
                    Debug.Log("Calling BuildBlock Server RPC ");
                    _world.BuildBlockServerRpc(hit.point - (ray.direction / 10000.0f));
                    /*localCoordinate = hit.point - (ray.direction / 10000.0f) - chunk.transform.position;
                    chunk.GetComponent<Chunk>().BuildBlockServerRpc(localCoordinate);
                    //_world.GetComponent<World>().UpdateMeshCollider(chunk);
                    Debug.Log("NOW SENDING RPC");
                    chunk.GetComponent<Chunk>().PingServerRpc("TESTRPC");*/
                    break;
                case RaycastAction.Shoot:
                    chunk = hit.transform.gameObject;
                    localCoordinate = hit.point + (ray.direction / 10000.0f) - chunk.transform.position;
                    chunk.GetComponent<Chunk>().DamageBlock(localCoordinate, 50);
                    //_world.GetComponent<World>().UpdateMeshCollider(chunk);
                    break;
                case RaycastAction.HighlightBlock:
                    float epsilon = 0.0001f;
                    var blockDim = _highlightBlock.transform.localScale;
                    float blockThickness = blockDim.x;
                    float blockSize = blockDim.y;
                    Vector3 coord = hit.point - (ray.direction / 10000.0f);
                    Vector3 blockPos = new Vector3(Mathf.FloorToInt(coord.x) + blockSize / 2, Mathf.FloorToInt(coord.y) + blockSize / 2, Mathf.FloorToInt(coord.z) + blockSize / 2);
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
            if (raycastAction == RaycastAction.HighlightBlock) {
                _highlightBlock.SetActive(false);
            }
        }
    }
}

public class MapPopup : MonoBehaviour {
    public enum MapPopupAction {
        Load, Save
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
            } else if (action == MapPopupAction.Save) {
                _world.SerializeChunks(mapName);
            }
            Close();
        }
        if (GUI.Button(new Rect(5, 80, windowRect.width - 10, 20), "Cancel")) {
            Close();
        }
    }
}
