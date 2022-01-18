using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.Netcode;
using UnityEngine;

public class Countdown : NetworkBehaviour {
    public NetworkVariable<int> networkTime = new NetworkVariable<int>(NetworkVariableReadPermission.Everyone);

    private float _startingTime; 
    [SerializeField] private float gameStartTime; 
    [SerializeField] private float respawnTime; 
    [SerializeField] private TMP_Text countdownNumbers;
    [SerializeField] private World world;
    [SerializeField] private TMP_Text countdownText;
    public bool countdownFinished = true;
    public bool localRespawn = false;
    
    public void StartCountdown(String message) {
        if (IsServer) {
            _startingTime = (int) gameStartTime;
            networkTime.Value = (int)gameStartTime;
        }
        
        networkTime.OnValueChanged += OnCountdownChanged;
        countdownFinished = false;
        countdownText.text = message;
        GetComponent<Canvas>().enabled = true;
    }
    
    public void StartLocalCountdown(String message) {
        _startingTime = respawnTime;
        localRespawn = true;
        countdownFinished = false;
        countdownText.text = message;
        GetComponent<Canvas>().enabled = true;
    }

    private void OnCountdownChanged(int oldValue, int newValue) {
        if (oldValue != newValue) {
            countdownNumbers.text = newValue.ToString("0");
            if (newValue <= 0) {
                StopCountdown();
            }
        }
    }

    // Update is called once per frame
    void Update() {
        if (countdownFinished) {
            return;
        }
        if (IsServer && !localRespawn) {
            _startingTime -= 1 * Time.deltaTime;
            networkTime.Value = (int)_startingTime;
        }
  
        if (localRespawn) {
            if (_startingTime <= 0) {
                StopCountdown();
            }
            _startingTime -= 1 * Time.deltaTime;
            countdownNumbers.text = _startingTime.ToString("0");
        }
        
    }

    void StopCountdown() {
        world.countdownFinished = true;
        countdownFinished = true;
        localRespawn = false;
        GetComponent<Canvas>().enabled = false;
    }
}
