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
        if (other.gameObject.GetComponent<Player>() == null || _world.respawnDirtyFlagState.Value)
            return;
        Player player = other.gameObject.GetComponent<Player>();
        if (_world.flagHolderId.Value == player.NetworkObjectId && player.team == Lobby.Team.Red)
            _world.OnFlagCaptureServerRpc(NetworkManager.Singleton.SpawnManager.GetLocalPlayerObject().NetworkObjectId);
    }
}
