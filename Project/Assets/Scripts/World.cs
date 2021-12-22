using System.Collections;
using System.Collections.Generic;
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
    
    [ServerRpc (RequireOwnership = false)]
    public void FindChunkServerRpc(Vector3 worldCoordinate) {
        Debug.Log("found chunk " + Mathf.FloorToInt((worldSize / 2 + worldCoordinate.x) / chunkSize) + " " + Mathf.FloorToInt((worldSize / 2 + worldCoordinate.z) / chunkSize));
        GameObject chunk = chunks[Mathf.Abs(Mathf.FloorToInt((worldSize / 2 + worldCoordinate.x) / chunkSize)), Mathf.Abs(Mathf.FloorToInt((worldSize / 2 + worldCoordinate.z) / chunkSize))];
        Vector3 localCoordinate = worldCoordinate - chunk.transform.position;
        chunk.GetComponent<Chunk>().BuildBlock(localCoordinate);
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
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                chunks[x,y].GetComponent<Chunk>().Load(mapName, x,y);
            }
        }
    }
}