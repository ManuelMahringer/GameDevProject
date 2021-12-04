using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;


// Credit: https://www.youtube.com/watch?v=NEUzB5vPYrE
public class Player : MonoBehaviour {
    [Header("Walk / Run Settings")] 
    public float walkSpeed = 3f;
    public float runSpeed = 5f;

    [Header("Jump Settings")]
    public float jumpForce = 10_000f;
    public ForceMode appliedForceMode = ForceMode.Force;

    [Header("Build/Destroy Settings")] 
    public float hitRange = 5f;

    [Header("Ground Tag Specification")]
    public String groundTag = "";
    public GameObject highlightBlock;

    [Header("Jumping State")]
    [SerializeField] private bool jump;
    [SerializeField] private bool isGrounded;
    
    [Header("Current Player Speed")]
    [SerializeField] private float currentSpeed;    

    private Rigidbody _rb;
    private Camera _camera;
    private RaycastHit _hit;
    private float _xAxis;
    private float _zAxis;
    private Vector3 _dxz;
    private Vector3 _groundLocation;
    private bool _respawn;

    private void Start() {
        _rb = GetComponent<Rigidbody>();
        _camera = GetComponentInChildren<Camera>();
    }

    private void Update() {
        _xAxis = Input.GetAxis("Horizontal");
        _zAxis = Input.GetAxis("Vertical");
        jump = Input.GetButton("Jump");
        _respawn = Input.GetKey(KeyCode.R);

        if (isGrounded) {
            // Update speed if grounded
            currentSpeed = Input.GetKey(KeyCode.LeftShift) ? runSpeed : walkSpeed;
        }

        SetGrounded();

        Vector3 midPoint = new Vector3(_camera.pixelWidth / 2, _camera.pixelHeight / 2);
        Ray ray = _camera.ScreenPointToRay(midPoint);
        if (Physics.Raycast(ray, out var hit, hitRange)) {
            Vector3 coord = hit.point - (ray.direction / 10000.0f);
            highlightBlock.SetActive(true);
            highlightBlock.transform.position = new Vector3(Mathf.FloorToInt(coord.x+0.5f), Mathf.FloorToInt(coord.y) + 0.5f, Mathf.FloorToInt(coord.z+0.5f));
        }
        else {
            highlightBlock.SetActive(false);
        }
    }
    
    private void FixedUpdate() {
        // Move
        if (isGrounded) { // update direction if grounded
            _dxz = transform.TransformDirection(_xAxis, 0f, _zAxis);
            _rb.velocity = Vector3.zero; // artificial friction when grounded
        }
        _rb.MovePosition(transform.position +
                         Vector3.ClampMagnitude(currentSpeed * _dxz * Time.deltaTime, currentSpeed));

        // Jump
        if (jump && isGrounded) {
            _rb.AddForce(jumpForce * _rb.mass * Time.deltaTime * Vector3.up, appliedForceMode);
            isGrounded = false;
        }
        
        // TODO: remove
        if (_respawn) {
            _rb.velocity = Vector3.zero;
            _rb.position = new Vector3(0, 10, 0);
        }
    }
    
    private void SetGrounded() {
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
}