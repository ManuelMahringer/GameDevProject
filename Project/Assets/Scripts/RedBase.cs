using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Netcode;
using UnityEngine;

public class RedBase : MonoBehaviour {
    private World _world;
    
    void Start() {
        _world = GameObject.Find("World").GetComponent<World>();
    }

    private void OnTriggerEnter(Collider other) {
        if (other.gameObject.GetComponent<Player>() == null)
            return;
        Player player = other.gameObject.GetComponent<Player>();
        if (player.hasFlag && player.team == Lobby.Team.Red)
            _world.OnFlagCaptureServerRpc(NetworkManager.Singleton.SpawnManager.GetLocalPlayerObject().NetworkObjectId);
    }
}
