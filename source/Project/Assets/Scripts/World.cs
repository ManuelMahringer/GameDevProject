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
    public int height;
    
    [SerializeField] public float chunkSize;
    [SerializeField] public int capturesToWin;

    [SerializeField] private GameObject flag;
    [SerializeField] private Vector3 initFlagPos = new Vector3(2, 3, 2);

    [SerializeField] public Vector3 baseRedPos;
    [SerializeField] public Vector3 baseBluePos;
    
    [SerializeField] private GameObject baseRed;
    [SerializeField] private GameObject baseBlue;

    [SerializeField] public string protectionLayerName;
    [SerializeField] public string baseLayerName;
    [SerializeField] public string borderLayerName;
    [SerializeField] private GameObject baseBlueProtectionZone;
    [SerializeField] private GameObject baseRedProtectionZone;
    [SerializeField] private GameObject flagProtectionZone;
    
    [SerializeField] public float statusMsgShowTime;
    
    public bool enableGenerate;

    [HideInInspector]
    public string selectedMap;
    [HideInInspector]
    public readonly NetworkVariable<bool> gameStarted = new NetworkVariable<bool>(NetworkVariableReadPermission.Everyone);
    [HideInInspector]
    public readonly NetworkVariable<int> redFlagCnt = new NetworkVariable<int>(NetworkVariableReadPermission.Everyone);
    [HideInInspector]
    public readonly NetworkVariable<int> blueFlagCnt = new NetworkVariable<int>(NetworkVariableReadPermission.Everyone);
    [HideInInspector]
    public readonly NetworkVariable<bool> gameEnded = new NetworkVariable<bool>(NetworkVariableReadPermission.Everyone);
    [HideInInspector]
    public readonly NetworkVariable<ulong> flagHolderId = new NetworkVariable<ulong>(NetworkVariableReadPermission.Everyone);
    [HideInInspector]
    public readonly NetworkVariable<bool> respawnDirtyFlagState = new NetworkVariable<bool>(NetworkVariableReadPermission.Everyone);
    [HideInInspector]
    public readonly NetworkVariable<ulong> transformBasePlayer = new NetworkVariable<ulong>(NetworkVariableReadPermission.Everyone);
    [HideInInspector]
    public readonly NetworkVariable<bool> hostQuit = new NetworkVariable<bool>(NetworkVariableReadPermission.Everyone);
    
    private GameObject[,,] _chunks;
    private float _worldSize;
    [NonSerialized]
    public bool countdownFinished;

    private void Start() {
        gameStarted.OnValueChanged += OnGameStarted;
        transformBasePlayer.OnValueChanged += OnPlayerInBase;
        countdownFinished = false;
    }

    private void OnGameStarted(bool oldVal, bool newVal) {
        flag.transform.position = initFlagPos;
        baseRed.transform.position = baseRedPos;
        baseBlue.transform.position = baseBluePos;
        baseRedProtectionZone.transform.position = baseRedPos;
        baseBlueProtectionZone.transform.position = baseBluePos;
        flagProtectionZone.transform.position = new Vector3(initFlagPos.x, initFlagPos.y - 0.5f, initFlagPos.z);
        flag.SetActive(true);
    }
    
    public bool InProtectedZone(Vector3 center) {
        // Ignore all layermasks but one: https://answers.unity.com/questions/1164722/raycast-ignore-layers-except.html
        return Physics.CheckBox(center, new Vector3(0.45f, 0.45f, 0.45f), Quaternion.identity, 1 << LayerMask.NameToLayer(protectionLayerName), QueryTriggerInteraction.Collide);
    }

    private void UpdateCaptureCounts(Lobby.Team team) {
        if (team == Lobby.Team.Red)
            redFlagCnt.Value += 1;
        else if (team == Lobby.Team.Blue)
            blueFlagCnt.Value += 1;
        if (blueFlagCnt.Value == capturesToWin || redFlagCnt.Value == capturesToWin)
            gameEnded.Value = true;
    }

    public void BuildWorld() {
        _worldSize = size * chunkSize;
        _chunks = new GameObject[size, height, size];
        // Instantiate chunks
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < height; y++) {
                for (int z = 0; z < size; z++) {
                    Debug.Log("instantiate now");
                    Debug.Log("Selected World " + selectedMap);
                    _chunks[x, y, z] = Instantiate(chunkPrefab, new Vector3(-_worldSize / 2 + chunkSize * x, -height * chunkSize / 2 + chunkSize * y, -_worldSize / 2 + chunkSize * z), Quaternion.identity); //  This quaternion corresponds to "no rotation" - the object is perfectly aligned with the world or parent axes.
                    _chunks[x,y, z].GetComponent<NetworkObject>().Spawn();
                }
            }
        }
    }

    public void SerializeChunks(string mapName) {
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < height; y++) {
                for (int z = 0; z < size; z++) {
                    _chunks[x, y, z].GetComponent<Chunk>().Serialize(mapName, x, y, z);
                }
            }
        }
    }

    public void LoadChunks(string mapName) {
        Debug.Log(_chunks);
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < height; y++) {
                for (int z = 0; z < size; z++) {
                    _chunks[x, y, z].GetComponent<Chunk>().Load(mapName, x, y, z);
                }
            }
        }
    }
    
    // RPC Calls
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
    
    [ServerRpc (RequireOwnership = false)]
    public void BuildBlockServerRpc(Vector3 worldCoordinate, BlockType blockType) {
        // Finds the correct chunk to build
        Debug.Log("BuildBlockServerRPC");
        int chunkX = Mathf.Abs(Mathf.FloorToInt((_worldSize / 2 + worldCoordinate.x) / chunkSize));
        int chunkY = Mathf.Abs(Mathf.FloorToInt((height * chunkSize / 2 + worldCoordinate.y) / chunkSize));
        int chunkZ = Mathf.Abs(Mathf.FloorToInt((_worldSize / 2 + worldCoordinate.z) / chunkSize));
        GameObject chunk = _chunks[chunkX, chunkY, chunkZ];
        Vector3 localCoordinate = worldCoordinate - chunk.transform.position;
        chunk.GetComponent<Chunk>().BuildBlockServer(localCoordinate, blockType);
    }
    
    [ServerRpc (RequireOwnership = false)]
    public void SetMapServerRpc(string map) {
        selectedMap = map;
        SetMapClientRpc(map);
    }
    
    [ClientRpc]
    private void SetMapClientRpc(string map) {
        selectedMap = map;
    }
    
    //
    // private void AddMeshCollider(int x, int z) {
    //     MeshCollider mc = _chunks[x, z, z].AddComponent<MeshCollider>();
    //     mc.material = worldMaterial;
    // }
    //
    // public void UpdateMeshCollider(GameObject chunk) {
    //     Destroy(chunk.GetComponent<MeshCollider>());
    //     MeshCollider mc = chunk.AddComponent<MeshCollider>();
    //     mc.material = worldMaterial;
    // }
    
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