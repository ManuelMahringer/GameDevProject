using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Player : MonoBehaviour
{

    private float speed = 6.0f;
    private CharacterController cc;

    void Start()
    {
        cc = GetComponent<CharacterController>();
    }

    void Update()
    {
        float dx = Input.GetAxis("Horizontal") * speed;
        float dz = Input.GetAxis("Vertical") * speed;

        Vector3 movement = new Vector3(dx * Time.deltaTime, 0, dz * Time.deltaTime);
        movement = Vector3.ClampMagnitude(movement, speed);
        movement = transform.TransformDirection(movement);
        cc.Move(movement);
    }
}
