using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Netcode;

public class GameNetworkManager : MonoBehaviour {
    
    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 10, 300, 300));
        if (!NetworkManager.Singleton.IsClient && !NetworkManager.Singleton.IsServer)
        {
            StartButtons();
        }
        else
        {
            StatusLabels();

            RebuildWorld();
        }

        GUILayout.EndArea();
    }

    static void StartButtons()
    {
        if (GUILayout.Button("Host")) {
            NetworkManager.Singleton.StartHost();
            GameObject.Find("World").GetComponent<World>().BuildWorld();
        }
        if (GUILayout.Button("Client")) NetworkManager.Singleton.StartClient();
        if (GUILayout.Button("Server")) {
            NetworkManager.Singleton.StartServer();
            GameObject.Find("World").GetComponent<World>().BuildWorld();
        }
    }

    static void StatusLabels()
    {
        var mode = NetworkManager.Singleton.IsHost ?
            "Host" : NetworkManager.Singleton.IsServer ? "Server" : "Client";

        GUILayout.Label("Transport: " +
                        NetworkManager.Singleton.NetworkConfig.NetworkTransport.GetType().Name);
        GUILayout.Label("Mode: " + mode);
    }

    static void RebuildWorld()
    {
        if (GUILayout.Button("Rebuild World"))
        {
            GameObject.Find("World").GetComponent<World>().ReBuildWorld();
        }
    }
}