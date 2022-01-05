using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using TMPro;
using UnityEngine;
using Unity.Netcode;
using UnityEngine.UI;

public class Lobby : NetworkBehaviour {
    [SerializeField]
    private TMP_Dropdown mapDropdown;
    private List<Map> maps;

    private World _world;
    
    // per default network variables can only be set by the server
    private NetworkVariable<int> _players = new NetworkVariable<int>(0);

    private NetworkString p0;
    private NetworkString p1;
    private NetworkString p2;
    private NetworkString p3;
    private NetworkString p4;
    private NetworkString p5;
    private int _playersOnClient = 0;
    
    private Dictionary<ulong, string> _clientNamesBlue = new Dictionary<ulong, string>();
    private Dictionary<ulong, string> _clientNamesRed = new Dictionary<ulong, string>();
    private TMP_Text[] _blueTMPTexts = new TMP_Text[3];
    private TMP_Text[] _redTMPTexts = new TMP_Text[3];
    private int _registered_count;


    public enum Team {
        Red,
        Blue
    }
    
    void Start() {
        _world = GameObject.Find("World").GetComponent<World>();
        
        Debug.Log("is HOst " +IsHost + " server" + IsServer );
        if (!IsHost) {
            GameObject.Find("ButtonStart").SetActive(false);
            mapDropdown.gameObject.SetActive(false);
        }

        GameObject.Find("internalClientID").GetComponent<Text>().text = NetworkManager.LocalClientId.ToString();

        //Debug.Log(GameObject.Find("Player0").GetComponentInChildren<TMP_Text>().text);
        _blueTMPTexts[0] = GameObject.Find("Player0").GetComponentInChildren<TMP_Text>();
        _blueTMPTexts[1] = GameObject.Find("Player1").GetComponentInChildren<TMP_Text>();
        _blueTMPTexts[2] = GameObject.Find("Player2").GetComponentInChildren<TMP_Text>();
        _redTMPTexts[0] = GameObject.Find("Player3").GetComponentInChildren<TMP_Text>();
        _redTMPTexts[1] = GameObject.Find("Player4").GetComponentInChildren<TMP_Text>();
        _redTMPTexts[2] = GameObject.Find("Player5").GetComponentInChildren<TMP_Text>();
        
        string[] mapPaths = Directory.GetDirectories(Application.persistentDataPath);
        string[] mapNames = mapPaths.Select(path => path.Split(Path.DirectorySeparatorChar).Last()).ToArray();

        maps = mapPaths.Zip(mapNames, (mapPath, mapName) => new Map {Path = mapPath, Name = mapName}).ToList();
        maps.Add(new Map {Name = "Generate", Path = ""}); // dummy option to still be able to generate the random map TODO: remove
        
        mapDropdown.ClearOptions();
        mapDropdown.AddOptions(mapNames.ToList());
        mapDropdown.AddOptions(new List<string> { "Generate" }); // dummy option to still be able to generate the random map TODO: remove
    }
    
    
    void OnPlayersChanged(int oldVal, int newVal) {
        if(oldVal != newVal)
            GameObject.Find("AmountOfPlayers").GetComponent<Text>().text = _players.Value.ToString();
    }

    void OnEnable() {
        _players.OnValueChanged += OnPlayersChanged;
    }

    // Update is called once per frame
    void Update() {
        if (IsServer)
            _players.Value = NetworkManager.ConnectedClientsIds.Count;
        
        if (_players.Value != _playersOnClient) {
            RequestCurrentLobbyServerRpc();
            _playersOnClient = _players.Value;
        }
    }
    
    [ServerRpc(RequireOwnership = false)]
    void RequestCurrentLobbyServerRpc() {
        UpdateNamesClientRpc(p0, p1, p2, p3, p4, p5);
    }
    
    [ClientRpc]
    public void UpdateNamesClientRpc(NetworkString p0, NetworkString p1,NetworkString p2,NetworkString p3,NetworkString p4,NetworkString p5) {
        _blueTMPTexts[0].text = p0.ToString();
        _blueTMPTexts[1].text = p1.ToString();
        _blueTMPTexts[2].text = p2.ToString();
        _redTMPTexts[0].text = p3.ToString();
        _redTMPTexts[1].text = p4.ToString();
        _redTMPTexts[2].text = p5.ToString();
    }


    [ServerRpc(RequireOwnership = false)]
    void AddPlayerServerRpc(Team team, ulong clientId, NetworkString ns) {
        Debug.Log("SERVERRPC CALLED - I HATE MY LIFE " + ns.ToString());
        if (team == Team.Blue) {
            if (_clientNamesBlue.ContainsKey(clientId)) {
                _clientNamesBlue[clientId] = ns.ToString();
            }
            else {
                _clientNamesBlue.Add(clientId, ns.ToString());
                switch (_clientNamesBlue.Count - 1) {
                    case 0:
                        p0 = ns;
                        break;
                    case 1:
                        p1 = ns;
                        break;
                    case 2:
                        p2 = ns;
                        break;
                }
            }
        } else if (team == Team.Red) {
            if (_clientNamesRed.ContainsKey(clientId)) {
                _clientNamesRed[clientId] = ns.ToString();
            }
            else {
                _clientNamesRed.Add(clientId, ns.ToString());
                switch (_clientNamesRed.Count - 1) {
                    case 0:
                        p3 = ns;
                        break;
                    case 1:
                        p4 = ns;
                        break;
                    case 2:
                        p5 = ns;
                        break;
                }
            }    
        }
        UpdateNamesClientRpc(p0, p1, p2, p3, p4, p5);
    }
    

    public void SubmitNameBlue() {
        AddPlayerServerRpc(Team.Blue, NetworkManager.LocalClientId, GameObject.Find("InputText").GetComponent<Text>().text);
    }
    public void SubmitNameRed() {
        AddPlayerServerRpc(Team.Red, NetworkManager.LocalClientId, GameObject.Find("InputText").GetComponent<Text>().text);
    }

    public void StartGame() {
        //ComponentManager.Map = maps[mapDropdown.value];
        _world.SetMapServerRpc(maps[mapDropdown.value].Name);
        _world.GetComponent<World>().BuildWorld();
        CloseLobbyClientRpc();
    }
    
    [ClientRpc]
    public void CloseLobbyClientRpc() {
        gameObject.SetActive(false);
    }
}