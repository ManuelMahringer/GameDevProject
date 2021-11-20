using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MouseLook : MonoBehaviour
{
    public enum RotationAxis
    {
        MouseXY = 0,
        MouseX = 1,
        MouseY = 2
    }

    public RotationAxis axis = RotationAxis.MouseXY;
    public float sensitivityHor = 8.0f;
    public float sensitivityVer = 8.0f;
    public float minVert = -20.0f;
    public float maxVert = 45.0f;
    private float _rotX = 0.0f;

    // Update is called once per frame
    void Update()
    {
        if(axis == RotationAxis.MouseX)
        {
            transform.Rotate(0, Input.GetAxis("Mouse X") * sensitivityHor, 0);
        }
        else if(axis == RotationAxis.MouseY)
        {
            _rotX -= Input.GetAxis("Mouse Y") * sensitivityVer;
            _rotX = Mathf.Clamp(_rotX, minVert, maxVert);
            float rotY = transform.localEulerAngles.y;
            transform.localEulerAngles = new Vector3(_rotX, rotY, 0);
        }
        else
        {
            _rotX -= Input.GetAxis("Mouse Y") * sensitivityVer;
            _rotX = Mathf.Clamp(_rotX, minVert, maxVert);
            float delta = Input.GetAxis("Mouse X") * sensitivityHor;
            float rotY = transform.localEulerAngles.y + delta;
            transform.localEulerAngles = new Vector3(_rotX, rotY, 0);
        }
    }
}
