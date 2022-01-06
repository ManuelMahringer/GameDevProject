using System.Collections;
using System.Collections.Generic;
using Unity.Netcode;
using UnityEngine;

public class AudioSync : NetworkBehaviour {
    private AudioSource source;

    public AudioClip[] clips;
    
    
    // Start is called before the first frame update
    void Start() {
        source = GetComponent<AudioSource>();
    }

    public void PlaySound(int id) {
        if (id >= 0 && id <= clips.Length) {
            SendSoundServerRpc(id);
        }
    }

    [ServerRpc(RequireOwnership = false)]
    public void SendSoundServerRpc(int id) {
        SendSoundClientRpc(id);
    }

    [ClientRpc]
    private void SendSoundClientRpc(int id) {
        source.PlayOneShot(clips[id]);
    }


}
