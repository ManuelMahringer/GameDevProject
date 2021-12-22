using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Serialization;
using UnityEngine.Networking;
using Unity.Netcode;


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
    private bool IsFalling => !isGrounded && _rb.velocity.y < 0;

    private float _health;
    private GameObject _world;
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
        _world = GameObject.Find("World");
        _rb = GetComponent<Rigidbody>();
        _audioSource = GetComponent<AudioSource>();
        _fallSound = Resources.Load("Sounds/hurt_fall") as AudioClip;
        _healthBar = GameObject.Find("HealthBar").GetComponent<Slider>();
        _health = maxHealth;
        _healthBar.maxValue = _health;
        _healthBar.value = _health;
        _highlightBlock = GameObject.Find("Highlight Slab");

        transform.position = new Vector3(0, 7, 0);
        
        Debug.Log("before is owner" + IsOwner + IsLocalPlayer );
        if (!IsOwner)
            return;
        Debug.Log(" after  is owner");
        _camera = gameObject.GetComponentInChildren<Camera>();
        _camera.enabled = true;
        Debug.Log("Camera enabled");
        gameObject.GetComponentInChildren<AudioListener>().enabled = true;
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
        bool destroyBlock = Input.GetMouseButtonDown(0);
        bool buildBlock = Input.GetMouseButtonDown(1);
        bool shoot = Input.GetKeyDown(KeyCode.T);
        bool rpc_call = Input.GetKeyDown(KeyCode.U);

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

        if (Input.GetKeyDown(KeyCode.Z)) {
            _world.GetComponent<World>().SerializeChunks();
        }

        if (Input.GetKeyDown(KeyCode.U)) {
            _world.GetComponent<World>().LoadChunks();
        }

        PerformRaycastAction(RaycastAction.HighlightBlock, hitRange);
    }

    private void FixedUpdate() {
        if (!IsLocalPlayer)
            return;
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
        _rb.MovePosition(transform.position + currentSpeed * _dxz * Time.deltaTime);

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
                    GameObject.Find("World").GetComponent<World>().FindChunkServerRpc(hit.point - (ray.direction / 10000.0f));
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
                    Vector3 blockPos = new Vector3(Mathf.FloorToInt(coord.x) + blockSize/2, Mathf.FloorToInt(coord.y) + blockSize/2, Mathf.FloorToInt(coord.z) + blockSize/2);
                    if (Math.Abs(hit.point.x - Mathf.Round(hit.point.x)) < epsilon) {  // looking at x face
                        _highlightBlock.transform.rotation = Quaternion.Euler(0, 0, 0);
                        blockPos.x += ray.direction.x > 0 ? blockSize / 2 - blockThickness / 2 : -blockSize / 2 + blockThickness / 2;
                    } else if (Math.Abs(hit.point.z - Mathf.Round(hit.point.z)) < epsilon) { // looking at z face
                        _highlightBlock.transform.rotation = Quaternion.Euler(0, 90, 0);
                        blockPos.z += ray.direction.z > 0 ? blockSize / 2 - blockThickness / 2 : -blockSize / 2 + blockThickness / 2;
                    } else {  // looking at y face
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