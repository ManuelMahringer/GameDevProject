using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Netcode;
using UnityEngine;

public class BlueBase : MonoBehaviour {
    private World _world;
    
    void Start() {
        _world = GameObject.Find("World").GetComponent<World>();
    }

    private void OnTriggerEnter(Collider other) {
        if (other.gameObject.GetComponent<Player>() != null && other.gameObject.GetComponent<Player>().hasFlag)
            _world.OnFlagCaptureServerRpc(NetworkManager.Singleton.SpawnManager.GetLocalPlayerObject().NetworkObjectId);        
    }
}
