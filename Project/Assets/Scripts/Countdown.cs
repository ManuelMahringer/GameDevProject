using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class Countdown : MonoBehaviour {
    private float currentTime = 0f;

    [SerializeField] private float startingTime; 
    [SerializeField] private TMP_Text countdownNumbers;
    [SerializeField] private World world;
    [SerializeField] private TMP_Text countdownText;
    public bool countdownFinished;


    
    public void StartCountdown(String message) {
        countdownFinished = false;
        countdownText.text = message;
        GetComponent<Canvas>().enabled = true;
        currentTime = startingTime;
    }

    // Update is called once per frame
    void Update() {
        if (countdownFinished) {
            return;
        }
        currentTime -= 1 * Time.deltaTime;
        countdownNumbers.text = currentTime.ToString("0");

        if (currentTime <= 0) {
            StopCountdown();
        }
    }

    void StopCountdown() {
        world.countdownFinished = true;
        countdownFinished = true;
        GetComponent<Canvas>().enabled = false;
    }
}
