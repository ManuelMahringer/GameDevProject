using Unity.Netcode;
using UnityEngine;

public class PlayersManager : NetworkBehaviour {
    NetworkVariable<int> playersInGame = new NetworkVariable<int>();

    public int PlayersInGame
    {
        get
        {
            return playersInGame.Value;
        }
    }

    void Start()
    {
        NetworkManager.Singleton.OnClientConnectedCallback += (id) =>
        {
            if (IsServer) {
                playersInGame.Value++; 
                Debug.Log("OnCliendConnectedCallback with value"+ playersInGame.Value);
            }
                
        };

        NetworkManager.Singleton.OnClientDisconnectCallback += (id) =>
        {
            if (IsServer) {
                playersInGame.Value--;
                Debug.Log("OnCliendConnectedCallback");
            }

        };
    }

  
        
}
