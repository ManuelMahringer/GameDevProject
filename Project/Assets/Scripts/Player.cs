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

    [Header("Ground Tag Specification")]
    public String groundTag = "";

    [Header("Jumping State")]
    [SerializeField] private bool jump;
    [SerializeField] private bool isGrounded;
    
    [Header("Current Player Speed")]
    [SerializeField] private float currentSpeed;    

    private Rigidbody _rb;
    private RaycastHit _hit;
    private float _xAxis;
    private float _zAxis;
    private Vector3 _dxz;
    private Vector3 _groundLocation;

    private void Start() {
        _rb = GetComponent<Rigidbody>();
    }

    private void Update() {
        _xAxis = Input.GetAxis("Horizontal");
        _zAxis = Input.GetAxis("Vertical");
        jump = Input.GetButton("Jump");

        if (isGrounded) { // Update speed if grounded
             currentSpeed = Input.GetKey(KeyCode.LeftShift) ? runSpeed : walkSpeed;
        }

        if (Physics.Raycast(transform.position, transform.TransformDirection(Vector3.down), out _hit, Mathf.Infinity)) {
            if (groundTag != null && groundTag != "") {
                if (String.Compare(_hit.collider.tag, groundTag, StringComparison.Ordinal) == 0) {
                    _groundLocation = _hit.point;
                }
            } else {
                _groundLocation = _hit.point;
            }

            float distanceFromPlayerToGround = Vector3.Distance(transform.position, _groundLocation);
            //Debug.Log(distanceFromPlayerToGround);
            isGrounded = !(distanceFromPlayerToGround > 1f + 0.0001f);
        }
        else {
            Debug.Log("Error in Player.cs: raycast should always hit an element underneath!");
        }
    }

    private void FixedUpdate() {
        // Move
        if (isGrounded) { // update direction if grounded
            _dxz = transform.TransformDirection(_xAxis, 0f, _zAxis);
        }
        _rb.MovePosition(transform.position +
                         Vector3.ClampMagnitude(currentSpeed * _dxz * Time.deltaTime, currentSpeed));
        
        // Jump
        if (jump && isGrounded) {
            _rb.AddForce(jumpForce * _rb.mass * Time.deltaTime * Vector3.up, appliedForceMode);
            isGrounded = false;
        }
    }
}