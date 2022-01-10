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

    public bool enableGenerate;
    public bool enableGameModeSelection;
    
    public string selectedMap;
    public NetworkVariable<bool> gameStarted = new NetworkVariable<bool>(NetworkVariableReadPermission.Everyone);
    public NetworkVariable<int> redFlagCnt = new NetworkVariable<int>(NetworkVariableReadPermission.Everyone);
    public NetworkVariable<int> blueFlagCnt = new NetworkVariable<int>(NetworkVariableReadPermission.Everyone);
    public NetworkVariable<bool> gameEnded = new NetworkVariable<bool>(NetworkVariableReadPermission.Everyone);
    public NetworkVariable<ulong> flagHolderId = new NetworkVariable<ulong>(NetworkVariableReadPermission.Everyone);
    public NetworkVariable<bool> respawnDirtyFlagState = new NetworkVariable<bool>(NetworkVariableReadPermission.Everyone);
    public NetworkVariable<ulong> transformBasePlayer = new NetworkVariable<ulong>(NetworkVariableReadPermission.Everyone);

    [SerializeField] public float chunkSize;

    [SerializeField] public int capturesToWin;
    
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
        transformBasePlayer.OnValueChanged += OnPlayerInBase;
    }

    private void OnGameStarted(bool oldVal, bool newVal) {
        flag.transform.position = initFlagPos;
        baseRed.transform.position = baseRedPos;
        baseBlue.transform.position = baseBluePos;
        flag.SetActive(true);
    }
    
    [ServerRpc (RequireOwnership = false)]
    public void OnFlagPickUpServerRpc(ulong playerId) {
        if (respawnDirtyFlagState.Value)
            return;
        Debug.Log("Server: Flag pickup from " + playerId + " at " + flag.transform.position);
        flagHolderId.Value = playerId;
        FlagPickupClientRpc(playerId);
    }
    
    [ClientRpc]
    private void FlagPickupClientRpc(ulong playerId) {
        flag.SetActive(false);
        Player flagHolder = GameNetworkManager.GetPlayerById(playerId);
        flagHolder.flag.SetActive(true);
    }


    [ServerRpc (RequireOwnership = false)]
    public void OnFlagCaptureServerRpc(ulong playerId) {
        if (respawnDirtyFlagState.Value)
            return;
        Player flagHolder = GameNetworkManager.GetPlayerById(playerId);
        Debug.Log("Server: Flag capture from " + playerId + ", team " + flagHolder.team);
        flagHolderId.Value = ulong.MaxValue;
        UpdateCaptureCounts(flagHolder.team);
        FlagCaptureClientRpc(playerId);
    }
    
    [ClientRpc]
    private void FlagCaptureClientRpc(ulong playerId) {
        flag.transform.position = initFlagPos;
        flag.SetActive(true);
        Player flagHolder = GameNetworkManager.GetPlayerById(playerId);
        flagHolder.flag.SetActive(false);
    }

    [ServerRpc(RequireOwnership = false)]
    public void DropFlagServerRpc(ulong playerId, Vector3 deathPos) {
        DropFlagClientRpc(playerId, deathPos);
    }

    [ClientRpc]
    private void DropFlagClientRpc(ulong playerId, Vector3 deathPos) {
        flag.transform.position = deathPos;
        flag.SetActive(true);
        Player flagHolder = GameNetworkManager.GetPlayerById(playerId);
        flagHolder.flag.SetActive(false);
    }
    
    [ServerRpc (RequireOwnership = false)]
    public void SetDirtyFlagStateServerRpc() {
        Debug.Log("Server: Setting dirty state");
        respawnDirtyFlagState.Value = true;
    }
    
    [ServerRpc (RequireOwnership = false)]
    public void PlayerResetCallbackServerRpc(ulong playerId) {
        Debug.Log("Server: transfrom base position callback from player " + playerId);
        transformBasePlayer.Value = playerId;
    }
    
    private void OnPlayerInBase(ulong oldVal, ulong newVal) {
        if (newVal == flagHolderId.Value) {
            Debug.Log("Resetting dirty flag state");
            flagHolderId.Value = ulong.MaxValue;
            respawnDirtyFlagState.Value = false;
        }
    }

    private void UpdateCaptureCounts(Lobby.Team team) {
        if (team == Lobby.Team.Red)
            redFlagCnt.Value += 1;
        else if (team == Lobby.Team.Blue)
            blueFlagCnt.Value += 1;
        if (blueFlagCnt.Value == capturesToWin || redFlagCnt.Value == capturesToWin)
            gameEnded.Value = true;
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