using System.Collections;
using System.Collections.Generic;
using Unity.Netcode;
using UnityEngine;

public class MouseLook : NetworkBehaviour {
    
    [Header("Sensitivity Settings")]
    public float sensitivityHor = 5.0f;
    public float sensitivityVer = 5.0f;
    
    [Header("Vertical angle range")]
    public float minVert = -90.0f;
    public float maxVert = 90.0f;

    public bool active;
    
    private Camera _playerCamera;
    private float _rotX;

    void Start() {
        if (!IsLocalPlayer) {
            return;
        }
        _playerCamera = GetComponentInChildren<Camera>();
        active = true;
        //Cursor.lockState = CursorLockMode.Locked;
        //Cursor.visible = false;
    }

    void Update() {
        if (!IsOwner || !active) {
            return;
        }
        // Rotate camera around x
        _rotX -= Input.GetAxis("Mouse Y") * sensitivityVer;
        _rotX = Mathf.Clamp(_rotX, minVert, maxVert);
        _playerCamera.transform.localEulerAngles = new Vector3(_rotX, 0, 0);
                
        // Rotate player object around y
        float rotY = Input.GetAxis("Mouse X") * sensitivityHor;
        transform.Rotate(0, rotY, 0);
    }
}