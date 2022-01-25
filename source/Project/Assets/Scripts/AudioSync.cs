using System.Collections;
using System.Collections.Generic;
using Unity.Netcode;
using UnityEngine;

public class AudioSync : NetworkBehaviour {
    public AudioClip[] clips;
    
    
    // Start is called before the first frame update
    void Start() {
        //source = GetComponent<AudioSource>();
    }

    public void PlaySound(int id) {
        if (id >= 0 && id <= clips.Length) {
            SendSoundServerRpc(id, NetworkObjectId);
        }
    }
    
    public void StartSoundLoop() {
        StartSoundLoopServerRpc(NetworkObjectId);
    }
    
    public void StopSoundLoop() {
        StopSoundLoopServerRpc(NetworkObjectId);
    }
    
    [ServerRpc(RequireOwnership = false)]
    public void SendSoundServerRpc(int id, ulong netid) {
        SendSoundClientRpc(id, netid);
    }
    
    [ServerRpc(RequireOwnership = false)]
    public void StartSoundLoopServerRpc(ulong netid) {
        SendSoundLoopClientRpc(netid);
    }
    
    [ServerRpc(RequireOwnership = false)]
    public void StopSoundLoopServerRpc(ulong netid) {
        StopSoundLoopClientRpc(netid);
    }

    [ClientRpc]
    private void SendSoundClientRpc(int id, ulong netid) {
        NetworkManager.SpawnManager.SpawnedObjects[netid].GetComponents<AudioSource>()[0].PlayOneShot(clips[id]);
        //Debug.Log("audio source "+ NetworkManager.SpawnManager.SpawnedObjects[netid]);
    }
    
    [ClientRpc]
    private void SendSoundLoopClientRpc(ulong netid) {
        NetworkManager.SpawnManager.SpawnedObjects[netid].GetComponents<AudioSource>()[1].Play();
    }
    
    [ClientRpc]
    private void StopSoundLoopClientRpc(ulong netid) {
        NetworkManager.SpawnManager.SpawnedObjects[netid].GetComponents<AudioSource>()[1].Stop();
    }
}
