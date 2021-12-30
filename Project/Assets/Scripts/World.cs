using System;
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
    
    public NetworkVariable<Vector3> testNetworkVar = new NetworkVariable<Vector3>(Vector3.zero);

    [ServerRpc (RequireOwnership = false)]
    public void GetInitialChunkDataServerRpc() {
        if (!IsServer)
            return;
        // foreach (var chunk in chunks) {
        //     var c = chunk.GetComponent<Chunk>();
        //     c.ReceiveInitialChunkDataClientRpc(c.FlattenBlocks());
        // }
        Debug.Log("SERVER: SENDING INITIAL CHUNK DATA");
        // for (int x = 0; x < size; x++) {
        //     for (int y = 0; y < size; y++) {
        //         var c = chunks[x, y].GetComponent<Chunk>();
        //         StartCoroutine(Test(c));
        //     }
        // }
        var c = chunks[0,0].GetComponent<Chunk>(); 
        c.ReceiveInitialChunkDataClientRpc(c.FlattenBlocks());
    }
    
    
    private IEnumerator Test(Chunk c) {
        yield return new WaitForSeconds(1);
        c.ReceiveInitialChunkDataClientRpc(c.FlattenBlocks());
    }

    [ServerRpc (RequireOwnership = false)]
    public void SetTestNetworkVarServerRpc(Vector3 v) {
        Debug.Log("Changed network variable by client request to " + v);
        testNetworkVar.Value = v;
    }

    [ServerRpc(RequireOwnership = false)]
    public void ReceiveTestVectorServerRpc(Vector3[] vs) {
        Debug.Log("SERVER RECEIVED VECTORS:");
        foreach (var v in vs) {
            Debug.Log(v);
        }
    }
    
    [ServerRpc (RequireOwnership = false)]
    public void BuildBlockServerRpc(Vector3 worldCoordinate) {
        //Debug.Log("found chunk " + Mathf.FloorToInt((worldSize / 2 + worldCoordinate.x) / chunkSize) + " " + Mathf.FloorToInt((worldSize / 2 + worldCoordinate.z) / chunkSize));
        // Finds the correct chunk to build
        if (!IsServer) {
            Debug.Log("ERROR: BuildBlockServerRpc executed on non-server");
            return;
        }
        Debug.Log("BuildBlockServerRPC");
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