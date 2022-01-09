using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using TMPro;
using UnityEngine;
using UnityEngine.Serialization;
using Unity.Netcode;
using UnityEngine.AI;

public class World : NetworkBehaviour {
    // Reference to the Chunk Prefab. Drag a Prefab into this field in the Inspector.
    [SerializeField] private GameObject chunkPrefab;
    public PhysicMaterial worldMaterial;
    public int size;

    public string selectedMap;
    public NetworkVariable<bool> gameStarted = new NetworkVariable<bool>(NetworkVariableReadPermission.Everyone);
    public NetworkVariable<int> redFlagCnt = new NetworkVariable<int>(NetworkVariableReadPermission.Everyone);
    public NetworkVariable<int> blueFlagCnt = new NetworkVariable<int>(NetworkVariableReadPermission.Everyone);

    [SerializeField] public float chunkSize;
    
    [SerializeField]
    private GameObject flag;
    
    [SerializeField]
    private Vector3 initFlagPos = new Vector3(2, 3, 2);

    [SerializeField] public Vector3 baseRedPos;
    [SerializeField] public Vector3 baseBluePos;
    
    [SerializeField] private GameObject baseRed;
    [SerializeField] private GameObject baseBlue;

    private GameObject[,] _chunks;
    private float _worldSize;
    public bool countdownFinished;

    private void Start() {
        gameStarted.OnValueChanged += OnGameStarted;
    }

    private void OnGameStarted(bool oldVal, bool newVal) {
        flag.transform.position = initFlagPos;
        baseRed.transform.position = baseRedPos;
        baseBlue.transform.position = baseBluePos;
        flag.SetActive(true);
    }

    [ServerRpc (RequireOwnership = false)]
    public void OnFlagPickUpServerRpc(ulong playerId) {
        Debug.Log("Flag pickup from " + playerId + " at " + flag.transform.position);
        FlagPickUpClientRpc(playerId);
    }

    [ClientRpc]
    private void FlagPickUpClientRpc(ulong playerId) {
        flag.SetActive(false);
        GameNetworkManager.players[playerId].player.PickUpFlag();
    }

    [ServerRpc (RequireOwnership = false)]
    public void OnFlagCaptureServerRpc(ulong playerId) {
        Debug.Log("Flag capture from " + playerId);
        Player player = GameNetworkManager.players[playerId].player;
        if (player.team == Lobby.Team.Red)
            redFlagCnt.Value += 1;
        else if (player.team == Lobby.Team.Blue)
            blueFlagCnt.Value += 1;
        FlagCaptureClientRpc(playerId);
    }

    [ClientRpc]
    private void FlagCaptureClientRpc(ulong playerId) {
        flag.transform.position = initFlagPos;
        flag.SetActive(true);
        GameNetworkManager.players[playerId].player.DropFlag();
    }

    [ServerRpc (RequireOwnership = false)]
    public void PlaceFlagServerRpc(Vector3 pos) {
        Debug.Log("Placing flag at " + pos);
        PlaceFlagClientRpc(pos);
    }

    [ClientRpc]
    private void PlaceFlagClientRpc(Vector3 pos) {
        Debug.Log("Client: Placing flag at (hopefully) death position: " + pos);
        flag.transform.position = pos;
        flag.SetActive(true);
    }
    
    [ServerRpc (RequireOwnership = false)]
    public void BuildBlockServerRpc(Vector3 worldCoordinate, BlockType blockType) {
        // Finds the correct chunk to build
        Debug.Log("BuildBlockServerRPC");
        int chunkX = Mathf.Abs(Mathf.FloorToInt((_worldSize / 2 + worldCoordinate.x) / chunkSize));
        int chunkZ = Mathf.Abs(Mathf.FloorToInt((_worldSize / 2 + worldCoordinate.z) / chunkSize));
        GameObject chunk = _chunks[chunkX, chunkZ];
        Vector3 localCoordinate = worldCoordinate - chunk.transform.position;
        chunk.GetComponent<Chunk>().BuildBlockServer(localCoordinate, blockType);
    }
    
    [ServerRpc (RequireOwnership = false)]
    public void SetMapServerRpc(string map) {
        selectedMap = map;
        SetMapClientRpc(map);
    }
    
    [ClientRpc]
    public void SetMapClientRpc(string map) {
        selectedMap = map;
    }

    public void BuildWorld() {
        _worldSize = size * chunkSize;
        _chunks = new GameObject[size, size];
        // Instantiate chunks
        for (int x = 0; x < size; x++) {
            for (int z = 0; z < size; z++) {
                Debug.Log("instantiate now");
                Debug.Log("Selected World " + selectedMap);
                _chunks[x, z] = Instantiate(chunkPrefab, new Vector3(-_worldSize / 2 + chunkSize * x, 1, -_worldSize / 2 + chunkSize * z), Quaternion.identity); //  This quaternion corresponds to "no rotation" - the object is perfectly aligned with the world or parent axes.
                _chunks[x, z].GetComponent<NetworkObject>().Spawn();
            }
        }
    }

    public void ReBuildWorld() {
        if (!IsOwner) {
            return;
        }
        _worldSize = size * chunkSize;
        // Instantiate chunks
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                Debug.Log("instantiate now");
                _chunks[x, y].GetComponent<NetworkObject>().Despawn();
                _chunks[x, y] = Instantiate(chunkPrefab, new Vector3(-_worldSize / 2 + chunkSize * x, 1, -_worldSize / 2 + chunkSize * y), Quaternion.identity); //  This quaternion corresponds to "no rotation" - the object is perfectly aligned with the world or parent axes.
                if (selectedMap != "Generate") { // dummy option to still be able to generate the random map TODO: remove
                    _chunks[x, y].GetComponent<Chunk>().Load(selectedMap, x, y);
                }
                _chunks[x, y].GetComponent<NetworkObject>().Spawn();
            }
        }
    }
    

    public void SerializeChunks(string mapName) {
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                _chunks[x,y].GetComponent<Chunk>().Serialize(mapName, x,y);
            }
        }
    }

    public void LoadChunks(string mapName) {
        Debug.Log(_chunks);
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                _chunks[x,y].GetComponent<Chunk>().Load(mapName, x,y);
            }
        }
    }
    
    private void AddMeshCollider(int x, int z) {
        MeshCollider mc = _chunks[x, z].AddComponent<MeshCollider>();
        mc.material = worldMaterial;
    }

    public void UpdateMeshCollider(GameObject chunk) {
        Destroy(chunk.GetComponent<MeshCollider>());
        MeshCollider mc = chunk.AddComponent<MeshCollider>();
        mc.material = worldMaterial;
    }
    
    // [ServerRpc (RequireOwnership = false)]
    // public void GetInitialChunkDataServerRpc() {
    //     Debug.Log("SERVER: SENDING INITIAL CHUNK DATA");
    //     for (int x = 0; x < size; x++) {
    //         for (int y = 0; y < size; y++) {
    //             var c = chunks[x, y].GetComponent<Chunk>();
    //             StartCoroutine(Test(c));
    //         }
    //     }
    //     // var c = chunks[0,0].GetComponent<Chunk>(); 
    //     // c.ReceiveInitialChunkDataClientRpc(c.FlattenBlocks());
    // }
    
    
    // private IEnumerator Test(Chunk c) {
    //     yield return new WaitForSeconds(1);
    //     c.ReceiveInitialChunkDataClientRpc(c.FlattenBlocks());
    // }
}