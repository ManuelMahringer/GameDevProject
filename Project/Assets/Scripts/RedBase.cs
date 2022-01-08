using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RedBase : MonoBehaviour {
    private World _world;
    void Start() {
        _world = GameObject.Find("World").GetComponent<World>();
    }

    private void OnTriggerEnter(Collider other) {
        _world.OnFlagCapture(other.gameObject.GetComponent<Player>());
    }
}
