using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Netcode;

public class GameNetworkManager : MonoBehaviour {

    public static readonly Dictionary<ulong, PlayerTeam> players = new Dictionary<ulong, PlayerTeam>();

    public struct PlayerTeam {
        public Player player;
        public Lobby.Team team;

        public PlayerTeam(Player player, Lobby.Team team) {
            this.player = player;
            this.team = team;
        }
    }

    private void Start() {
        Debug.Log("HEY ITS THE GAME NETWORK MANAGER" + ComponentManager.mode.ToString());

        if (ComponentManager.mode == Mode.Client) {
            NetworkManager.Singleton.StartClient();
        }

        if (ComponentManager.mode == Mode.Server) {
            NetworkManager.Singleton.StartServer();
            //GameObject.Find("World").GetComponent<World>().BuildWorld();
        }

        if (ComponentManager.mode == Mode.Host) {
            Debug.Log("starting host");
            NetworkManager.Singleton.StartHost();
            //GameObject.Find("World").GetComponent<World>().BuildWorld();
        }
    }

    public static void RegisterPlayer(ulong netId, Player player, Lobby.Team team) {
        players.Add(netId, new PlayerTeam(player, team));
        player.transform.name = "Player " + netId;
        Debug.Log("Registered player " + netId);
    }

    public static void UnregisterPlayer(ulong netId) {
        players.Remove(netId);
        Debug.Log("Unregistered player: " + netId);
    }

    public static Player GetPlayerById(ulong netId) {
        return players[netId].player;
    }

    void OnGUI() {
        GUILayout.BeginArea(new Rect(10, 10, 300, 300));
        if (!NetworkManager.Singleton.IsClient && !NetworkManager.Singleton.IsServer) {
            //StartButtons();
        }
        else {
            //StatusLabels();

            //RebuildWorld();
        }

        GUILayout.EndArea();
    }
}