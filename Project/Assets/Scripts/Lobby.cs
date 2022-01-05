using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using TMPro;
using UnityEngine;
using Unity.Netcode;
using UnityEngine.UI;

public class Lobby : NetworkBehaviour {
    // per default network variables can only be set by the server
    private NetworkVariable<int> _players = new NetworkVariable<int>(0);
    //private NetworkVariable<int> _names = new NetworkVariable<int>[6]; 
    //private NetworkVariable<int> _player0 = new NetworkVariable<int> Everyone, 5);

    //private NetworkVariable<int> _testname = new NetworkVariable<int>();
    //private Dictionary<ulong, NetworkString> _clientNames = new Dictionary<ulong,NetworkString>();
    private Dictionary<ulong, int> _clientIds;
    private TMP_Text[] _playerstrings = new TMP_Text[6];
    private int _registered_count;
        
    
    // Start is called before the first frame update
    void Start()
    {
        if (IsServer) {
            _clientIds = new Dictionary<ulong,int>();
            _registered_count = 1; // me - the host
        }
        
       GameObject.Find("internalClientID").GetComponent<Text>().text = NetworkManager.LocalClientId.ToString();
       Debug.Log("MAAAAAAAAAANI"+NetworkManager.LocalClientId.ToString());

       Debug.Log(GameObject.Find("Player0").GetComponentInChildren<TMP_Text>().text);
       _playerstrings[0] = GameObject.Find("Player0").GetComponentInChildren<TMP_Text>();
       _playerstrings[1] = GameObject.Find("Player1").GetComponentInChildren<TMP_Text>();
       _playerstrings[2] = GameObject.Find("Player2").GetComponentInChildren<TMP_Text>();
       _playerstrings[3] = GameObject.Find("Player3").GetComponentInChildren<TMP_Text>();
       _playerstrings[4] = GameObject.Find("Player4").GetComponentInChildren<TMP_Text>();
       _playerstrings[5] = GameObject.Find("Player5").GetComponentInChildren<TMP_Text>();

       
    }
    
    void OnEnable()
    {
        // Subscribe for when amount of players value changes
        _players.OnValueChanged += OnPlayersChanged;
        /*
        for (int i = 0; i < _names.Length; i++) {
            _names[i]  = new NetworkVariable<NetworkString>("asdf");
            _names[i].OnValueChanged+= OnNameChanged;
        }*/
        //_testname.OnValueChanged += OnNameChanged;
    }

    // Update is called once per frame
    void Update() {
        if (IsServer)
            _players.Value = NetworkManager.ConnectedClientsIds.Count;
        
        //Debug.Log("Start printing players");
        
        /*foreach (KeyValuePair<ulong, NetworkClient> kvp in NetworkManager.ConnectedClients)
        {
            //textBox3.Text += ("Key = {0}, Value = {1}", kvp.Key, kvp.Value);
            Debug.Log("Key = " + kvp.Key + "  " + "Value = " +kvp.Value.ClientId);
        }*/
        //UpdateNames();
        
    }

    /*
    void OnNameChanged(NetworkString oldVal, NetworkString newVal) {
        Debug.Log("OnNameChangedCalled " + oldVal.ToString() + " " + newVal.ToString());
        if (oldVal.ToString() != newVal.ToString())
            UpdateNames();
    }*/

    void OnPlayersChanged(int oldVal, int newVal) {
        Debug.Log("On Players Changed called");
        if(oldVal != newVal)
            GameObject.Find("AmountOfPlayers").GetComponent<Text>().text = _players.Value.ToString();
    }
    
    /*
    [ServerRpc(RequireOwnership = false)]
    void AddPlayerServerRpc(Team team, ulong clientId, NetworkString ns) {
        Debug.Log("SERVERRPC CALLED - I HATE MY LIFE " + ns.ToString());
        if (_clientNames.ContainsKey(clientId)) {
            _clientNames[clientId] =  ns;
            
        }
        else {
            _clientNames.Add(clientId , ns);
            Debug.Log("Client Names count  " + _clientNames.Count );
            _names[_clientNames.Count-1].Value = ns.ToString();
            //_testname.Value = int.Parse(ns);
        }

        //UpdateNamesClientRpc();
        Debug.Log("names 2 changed");
    }
    
    
    [ServerRpc(RequireOwnership = false)]
    void AddPlayerServerRpc(Team team, ulong clientId) {
        Debug.Log("SERVERRPC CALLED - I HATE MY LIFE ");
        if (!_clientIds.ContainsKey(clientId)) {
            _clientIds[clientId] = _registered_count++; 
        }
    }
    /*
    [ClientRpc]
    public void UpdateNamesClientRpc() {
        for (int i = 0; i < _playerstrings.Length; i++) {
            Debug.Log("update names on client rpc  called with name " + i + " "  + _names[i].Value.ToString());
            _playerstrings[i].text = _names[i].Value.ToString();

        }
    }
    
    public void UpdateNames() {
        for (int i = 0; i < _playerstrings.Length; i++) {
            //Debug.Log("Value of testname "+ _testname.Value.ToString());
            //_playerstrings[i].text = _testname.Value.ToString();
            Debug.Log("Name of " + i + " "  + _names[i].Value);
            if (_names[i] != null) {
                _playerstrings[i].text = _names[i].Value.ToString();
            }
        }
    }

    public void SubmitName() {
        Debug.Log("submit name called with Input text "+ GameObject.Find("InputText").GetComponent<Text>().text);
        
        //Debug.Log("AAA"+GameObject.Find("InputText").GetComponent<Text>().text);
        //AddPlayerServerRpc(Team.Blue,NetworkManager.LocalClientId, GameObject.Find("InputText").GetComponent<Text>().text);
        AddPlayerServerRpc(Team.Blue,NetworkManager.LocalClientId);
    }

    public enum Team {
        Red,
        Blue
    }*/
}