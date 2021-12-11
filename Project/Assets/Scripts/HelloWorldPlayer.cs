using Unity.Netcode;
using UnityEngine;


public class HelloWorldPlayer : NetworkBehaviour {
    public NetworkVariable<Vector3> Position = new NetworkVariable<Vector3>();

    public override void OnNetworkSpawn() {
        transform.position = new Vector3(0, 2, 0);
        if (IsOwner) {
            Move();
        }
    }

    public void Move() {
        if (NetworkManager.Singleton.IsServer) {
            //var randomPosition = GetRandomPositionOnPlane();
            //transform.position = randomPosition;
            //Position.Value = randomPosition;
        }
        else {
            SubmitPositionRequestServerRpc();
        }
    }

    [ServerRpc]
    void SubmitPositionRequestServerRpc(ServerRpcParams rpcParams = default) {
        Position.Value = transform.position;
        
    }

    static Vector3 GetRandomPositionOnPlane() {
        return new Vector3(Random.Range(-3f, 3f), 1f, Random.Range(-3f, 3f));
    }

    void Update() {
        //transform.position = Position.Value;
    }
}
