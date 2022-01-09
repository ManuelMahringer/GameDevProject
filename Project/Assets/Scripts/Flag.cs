using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Netcode;
using UnityEngine;

public class Flag : NetworkBehaviour {
    private World _world;
    
    void Start() {
        _world = GameObject.Find("World").GetComponent<World>();
    }

    private void OnTriggerEnter(Collider other) {
        _world.OnFlagPickUp(other.gameObject.GetComponent<Player>());
    }
}
