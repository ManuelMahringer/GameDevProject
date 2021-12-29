using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using UnityEngine.Serialization;
using Unity.Netcode;

public class World : NetworkBehaviour {
    // Reference to the Chunk Prefab. Drag a Prefab into this field in the Inspector.
    [SerializeField] private GameObject chunkPrefab;
    public PhysicMaterial worldMaterial;
    public int size;
    
    [SerializeField] private float chunkSize;

    private GameObject[,] chunks;
    private float worldSize;
    
    public NetworkVariable<char> testNetworkVar = new NetworkVariable<char>('H');

    [ServerRpc (RequireOwnership = false)]
    public void SetTestNetworkVarServerRpc(char c) {
        Debug.Log("Changed network variable by client request to " + c);
        testNetworkVar.Value = c;
    }
    
    [ServerRpc (RequireOwnership = false)]
    public void BuildBlockServerRpc(Vector3 worldCoordinate) {
        //Debug.Log("found chunk " + Mathf.FloorToInt((worldSize / 2 + worldCoordinate.x) / chunkSize) + " " + Mathf.FloorToInt((worldSize / 2 + worldCoordinate.z) / chunkSize));
        // Finds the correct chunk to build
        int chunkX = Mathf.Abs(Mathf.FloorToInt((worldSize / 2 + worldCoordinate.x) / chunkSize));
        int chunkZ = Mathf.Abs(Mathf.FloorToInt((worldSize / 2 + worldCoordinate.z) / chunkSize));
        GameObject chunk = chunks[chunkX, chunkZ];
        Vector3 localCoordinate = worldCoordinate - chunk.transform.position;
        chunk.GetComponent<Chunk>().BuildBlockServer(localCoordinate);
    }
    
    public void BuildWorld() {
        worldSize = size * chunkSize;
        chunks = new GameObject[size, size];
        // Instantiate chunks
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                Debug.Log("instantiate now");
                chunks[x, y] = Instantiate(chunkPrefab, new Vector3(-worldSize / 2 + chunkSize * x, 1, -worldSize / 2 + chunkSize * y), Quaternion.identity); //  This quaternion corresponds to "no rotation" - the object is perfectly aligned with the world or parent axes.
                if (ComponentManager.Map.Name != "Generate") { // dummy option to still be able to generate the random map TODO: remove
                    chunks[x, y].GetComponent<Chunk>().Load(ComponentManager.Map.Name, x, y);
                }
                chunks[x, y].GetComponent<NetworkObject>().Spawn();
                //Debug.Log("spawned");
                //AddMeshCollider(x, y);
            }
        }
    }
    
    public void ReBuildWorld() {
        if (!IsOwner) {
            return;
        }
        worldSize = size * chunkSize;
        // Instantiate chunks
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                Debug.Log("instantiate now");
                chunks[x, y].GetComponent<NetworkObject>().Despawn();
                chunks[x, y] = Instantiate(chunkPrefab, new Vector3(-worldSize / 2 + chunkSize * x, 1, -worldSize / 2 + chunkSize * y), Quaternion.identity); //  This quaternion corresponds to "no rotation" - the object is perfectly aligned with the world or parent axes.
                if (ComponentManager.Map.Name != "Generate") { // dummy option to still be able to generate the random map TODO: remove
                    chunks[x, y].GetComponent<Chunk>().Load(ComponentManager.Map.Name, x, y);
                }
                chunks[x, y].GetComponent<NetworkObject>().Spawn();
                //Debug.Log("spawned");
                //AddMeshCollider(x, y);
            }
        }
    }

    public void SerializeChunks(string mapName) {
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                chunks[x,y].GetComponent<Chunk>().Serialize(mapName, x,y);
            }
        }
    }

    public void LoadChunks(string mapName) {
        Debug.Log(chunks);
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                chunks[x,y].GetComponent<Chunk>().Load(mapName, x,y);
            }
        }
    }
    
    private void AddMeshCollider(int x, int z) {
        MeshCollider mc = chunks[x, z].AddComponent<MeshCollider>();
        mc.material = worldMaterial;
    }

    public void UpdateMeshCollider(GameObject chunk) {
        Destroy(chunk.GetComponent<MeshCollider>());
        MeshCollider mc = chunk.AddComponent<MeshCollider>();
        mc.material = worldMaterial;
    }
}

class MyNetworkVariable : NetworkVariableBase {
    public override void WriteDelta(FastBufferWriter writer) {
        throw new System.NotImplementedException();
    }

    public override void WriteField(FastBufferWriter writer) {
        throw new System.NotImplementedException();
    }

    public override void ReadField(FastBufferReader reader) {
        throw new System.NotImplementedException();
    }

    public override void ReadDelta(FastBufferReader reader, bool keepDirtyDelta) {
        throw new System.NotImplementedException();
    }
}