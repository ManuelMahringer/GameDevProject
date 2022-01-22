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
    private TMP_Text _errortext;
    private Text _nameInputText;


    private GameObject _joinRed;
    private GameObject _joinBlue;
    private GameObject _inputField;
    private GameObject _infoTextName;
    private GameObject _worldBorders;
    private void DisableControls() {
        if (_joinBlue.activeSelf) {
            _joinBlue.SetActive(false);
        }
        if (_joinRed.activeSelf) {
            _joinRed.SetActive(false);
        }
        _inputField.SetActive(false);
        _infoTextName.SetActive(false);
    }


    public enum Team {
        Red,
        Blue
    }
    
    void Start() {
        _world = GameObject.Find("World").GetComponent<World>();
        _errortext = GameObject.Find("Errortext").GetComponent<TMP_Text>();
        _nameInputText = GameObject.Find("InputText").GetComponent<Text>();
        _joinRed = GameObject.Find("JoinRedButton");
        _joinBlue = GameObject.Find("JoinBlueButton");
        _inputField = GameObject.Find("InputField");
        _infoTextName = GameObject.Find("Name");
        _worldBorders = GameObject.Find("WorldBorders/Bottom_Plane");
        
        Debug.Log("is HOst " +IsHost + " server" + IsServer );
        if (!IsHost) {
            GameObject.Find("ButtonStart").SetActive(false);
            mapDropdown.gameObject.SetActive(false);
        }

        //GameObject.Find("internalClientID").GetComponent<Text>().text = NetworkManager.LocalClientId.ToString();

        //Debug.Log(GameObject.Find("Player0").GetComponentInChildren<TMP_Text>().text);
        _blueTMPTexts[0] = GameObject.Find("Player0").GetComponentInChildren<TMP_Text>();
        _blueTMPTexts[1] = GameObject.Find("Player1").GetComponentInChildren<TMP_Text>();
        _blueTMPTexts[2] = GameObject.Find("Player2").GetComponentInChildren<TMP_Text>();
        _redTMPTexts[0] = GameObject.Find("Player3").GetComponentInChildren<TMP_Text>();
        _redTMPTexts[1] = GameObject.Find("Player4").GetComponentInChildren<TMP_Text>();
        _redTMPTexts[2] = GameObject.Find("Player5").GetComponentInChildren<TMP_Text>();
        
        string[] mapPaths = Directory.GetDirectories(System.IO.Directory.GetCurrentDirectory() + Path.DirectorySeparatorChar + "Maps");
        string[] mapNames = mapPaths.Select(path => path.Split(Path.DirectorySeparatorChar).Last()).ToArray();

        maps = mapPaths.Zip(mapNames, (mapPath, mapName) => new Map {Path = mapPath, Name = mapName}).ToList();

        maps.Add(new Map {Name = "Generate", Path = ""}); // dummy option to still be able to generate the random map TODO: remove
        
        mapDropdown.ClearOptions();
        mapDropdown.AddOptions(mapNames.ToList());
        if (_world.enableGenerate)
            mapDropdown.AddOptions(new List<string> { "Generate" }); // dummy option to still be able to generate the random map TODO: remove
    }
    
    
    // void OnPlayersChanged(int oldVal, int newVal) {
    //     if(oldVal != newVal)
    //         GameObject.Find("AmountOfPlayers").GetComponent<Text>().text = _players.Value.ToString();
    // }

    // void OnEnable() {
    //     _players.OnValueChanged += OnPlayersChanged;
    // }

    // Update is called once per frame
    void Update() {
        if (IsServer)
            _players.Value = NetworkManager.ConnectedClientsIds.Count;
        
        if (_players.Value != _playersOnClient) {
            RequestCurrentLobbyServerRpc();
            _playersOnClient = _players.Value;

        }

        if (_clientNamesBlue.Count >= 3) {
            _joinBlue.SetActive(false);
        }
        if (_clientNamesRed.Count >= 3) {
            _joinBlue.SetActive(false);
        }
    }
    
    [ServerRpc(RequireOwnership = false)]
    void RequestCurrentLobbyServerRpc() {
        UpdateNamesClientRpc(p0, p1, p2, p3, p4, p5);
        UpdateJoinButtonsClientRpc(_clientNamesBlue.Count < 3, _clientNamesRed.Count < 3);
    }
    
    [ClientRpc]
    public void UpdateNamesClientRpc(NetworkString p0, NetworkString p1,NetworkString p2,NetworkString p3,NetworkString p4,NetworkString p5) {
        Debug.Log("update names client rpc ");
        _blueTMPTexts[0].text = p0.ToString();
        _blueTMPTexts[1].text = p1.ToString();
        _blueTMPTexts[2].text = p2.ToString();
        _redTMPTexts[0].text = p3.ToString();
        _redTMPTexts[1].text = p4.ToString();
        _redTMPTexts[2].text = p5.ToString();
        Debug.Log("calling deactivate Join");
        
    }

    [ClientRpc]
    public void UpdateJoinButtonsClientRpc(bool blue, bool red) {
        if (!blue) {
            _joinBlue.SetActive(false);

        }
        if (!red) {
            _joinRed.SetActive(false);

        }
    }
    
    
    [ServerRpc(RequireOwnership = false)]
    void AddPlayerServerRpc(Team team, ulong clientId, NetworkString ns) {
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
        if (String.IsNullOrEmpty(_nameInputText.text)) {
            _errortext.text = "Please enter your name before joining a team!";
            return;
        }
        _errortext.text = "";
        
        NetworkObject player = NetworkManager.Singleton.SpawnManager.GetLocalPlayerObject(); // NetworkManager.Singleton.LocalClient is null for some reason: https://forum.unity.com/threads/networkmanager-singleton-localclient-for-finding-the-local-player.1196902/
        if (string.IsNullOrEmpty(_blueTMPTexts[0].text)) {
            player.GetComponent<Player>().spawnOffset = -1;
        }
        else if(string.IsNullOrEmpty(_blueTMPTexts[1].text))
            player.GetComponent<Player>().spawnOffset = 0;
        else {
            player.GetComponent<Player>().spawnOffset = 1;
        }
        //player.GetComponent<Player>().spawnOffset = (_clientNamesBlue.Count-1)*5;
        AddPlayerServerRpc(Team.Blue, player.NetworkObjectId, _nameInputText.text);
        player.GetComponent<Player>().team = Team.Blue;
        player.GetComponent<Player>().playerName = _nameInputText.text;
        DisableControls();
    }
    public void SubmitNameRed() {
        if (String.IsNullOrEmpty(_nameInputText.text)) {
            _errortext.text = "Please enter your name before joining a team!";
            return;
        }
        _errortext.text = "";
        
        NetworkObject player = NetworkManager.Singleton.SpawnManager.GetLocalPlayerObject(); // NetworkManager.Singleton.LocalClient is null for some reason: https://forum.unity.com/threads/networkmanager-singleton-localclient-for-finding-the-local-player.1196902/
        if(string.IsNullOrEmpty(_redTMPTexts[0].text))
            player.GetComponent<Player>().spawnOffset = -1;
        else if(string.IsNullOrEmpty(_redTMPTexts[1].text))
            player.GetComponent<Player>().spawnOffset = 0;
        else {
            player.GetComponent<Player>().spawnOffset = 1;
        }
        AddPlayerServerRpc(Team.Red, player.NetworkObjectId, _nameInputText.text);
        player.GetComponent<Player>().team = Team.Red;
        player.GetComponent<Player>().playerName = _nameInputText.text;
        DisableControls();
    }

    public void StartGame() {
        if (_clientNamesBlue.Count != _clientNamesRed.Count || _players.Value <= 1) {
            _errortext.text =
                "Teams are not balanced! \nPlease make sure there is at least 1 player on each team and an equal amount of players on each team.";
            //return;
        }
        _world.SetMapServerRpc(maps[mapDropdown.value].Name);
        _world.gameStarted.Value = true;
        _world.GetComponent<World>().BuildWorld();
        CloseLobbyClientRpc();
    }
    
    [ClientRpc]
    private void CloseLobbyClientRpc() {
        // make bottom plane visible
        _worldBorders.GetComponent<MeshRenderer>().enabled = true;
        // disable Lobby overlay
        GameObject.Find("HUD (Countdown)").GetComponent<Countdown>().StartCountdown("Game starting in ...");
        gameObject.SetActive(false);
    }
}