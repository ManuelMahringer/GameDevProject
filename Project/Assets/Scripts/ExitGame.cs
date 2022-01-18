using System.Collections;
using System.Collections.Generic;
using Unity.Netcode;
using UnityEngine;

public class ExitGame : NetworkBehaviour {

    private World _world;

    private void Start() {
        _world = GameObject.Find("World").GetComponent<World>();
    }
    
    public void DoExitGame() {
        if (IsHost) {
            Debug.Log("Setting host quit");
            _world.hostQuit.Value = true;
        }
        Application.Quit();
    }
}